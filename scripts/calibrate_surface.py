"""Calibrate density, covariance overlap and global opacity on analytic patches.

Selection uses coverage/leakage constraints, detail error, depth error and point
cost. It never reads any of the four sk_00918 evaluation images.
"""
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion,binary_dilation
from scipy.spatial.transform import Rotation
from PIL import Image

from render_gaussians import configure_cuda
from ablate_render_conventions import analytic
from mesh_to_gs.surface import charts,sample_component,covariance,texture_colors
from mesh_to_gs.adaptive import TextureFeatures,prepare_importance
from mesh_to_gs.color import srgb_to_linear,linear_to_srgb


def texture_image(checker):
    yy,xx=np.mgrid[:512,:512]
    image=np.full((512,512,3),128,np.uint8)
    value=(((xx*checker//512)+(yy*checker//512))%2)*255
    image[:,256:]=value[:,256:,None]
    return image


def layer(angle=0,center=(0,0,3),extent=(1,1),color=(.5,.5,.5),texture=None):
    R=Rotation.from_euler('y',angle,degrees=True).as_matrix()
    v=np.array([[-extent[0],-extent[1],0],[extent[0],-extent[1],0],[-extent[0],extent[1],0],[extent[0],extent[1],0]])
    ids=np.array([[0,1,2],[1,3,2]])
    uv=np.array([[0.,0],[1,0],[0,1],[1,1]])
    surface=charts((v@R.T+center)[ids],ids,uv[ids] if texture is not None else None)
    return dict(surface=surface,R=R,center=np.array(center),extent=extent,checker=0,color=color,texture=texture)


def datasets(held_out=False):
    texture=TextureFeatures(texture_image(24 if held_out else 32))
    return [('inclined',[layer(55 if held_out else 75)]),
            ('detail',[layer(texture=texture)]),
            ('occlusion',[layer(0,(0,0,3.5),(1.5,1.2),(0,0,1)),layer(35 if held_out else 25,(-.35,0,2.5),(.45,.7),(1,0,0))])]


def sample_layers(layers,spacing,texture_weight):
    samples=[]
    for item in layers:
        s=item['surface'];feature=item['texture']
        prepare_importance(s,feature,spacing,texture_weight,1.)
        mass=float(np.sum(s.areas*s.importance))+.25*spacing*float(s.perimeter.sum())
        count=int(round(mass/spacing**2))+1
        p,n,uv,edges,h,nb=sample_component(s,0,count,123,feature)
        colors=texture_colors(feature.image,uv,linear=True) if feature is not None else np.tile(item['color'],(count,1))
        samples.append(dict(p=p,n=n,uv=uv,edges=edges,h=h,colors=colors))
    return samples


def shapes(samples,overlap):
    out=[]
    for s in samples:
        sc,q,_=covariance(s['p'],s['n'],s['edges'],s['h'],overlap,
             max_normal_rotation=.22,directional_support=True)
        out.append(dict(**s,sc=sc,q=q))
    return out


def render(items,opacity,K,width,height,reference):
    import torch
    from gsplat import rasterization
    def gpu(a):return torch.as_tensor(np.asarray(a),device='cuda',dtype=torch.float32)
    with torch.no_grad():
        image,alpha,_=rasterization(means=gpu(np.concatenate([s['p'] for s in items])),quats=gpu(np.concatenate([s['q'] for s in items])),
            scales=gpu(np.concatenate([s['sc'] for s in items])),colors=srgb_to_linear(gpu(np.concatenate([s['colors'] for s in items]))),
            opacities=gpu(np.full(sum(len(s['p']) for s in items),opacity)),viewmats=gpu(np.eye(4)[None]),Ks=gpu(K[None]),width=width,height=height,
            packed=True,render_mode='RGB+ED',eps2d=.3)
        rgb=linear_to_srgb(image[0,:,:,:3]+1-alpha[0]).clip(0,1).cpu().numpy()
        d=image[0,:,:,3].cpu().numpy();a=alpha[0,:,:,0].cpu().numpy()
    ref,ra,rd=reference
    valid=binary_erosion(ra>.999,iterations=5)&(rd>0)
    outside=(ra<1e-4)&binary_dilation(ra>.5,iterations=5)
    mse=np.mean((rgb[valid]-ref[valid])**2)
    return dict(points=sum(len(s['p']) for s in items),coverage=float((a[valid]>=.95).mean()),mean_alpha=float(a[valid].mean()),
                depth_mae=float(np.abs(d-rd)[valid].mean()),psnr_db=float(-10*np.log10(max(mse,1e-12))),
                leakage=float(a[outside].mean()) if outside.any() else 0.),rgb


def aggregate(rows):
    result={k:float(np.mean([r[k] for r in rows])) for k in ['points','coverage','mean_alpha','depth_mae','leakage']}
    result['psnr_db']=float(-10*np.log10(np.mean([10**(-r['psnr_db']/10) for r in rows])))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/surface_calibration'))
    parser.add_argument('--objective',choices=('quality','balanced'),default='quality',
                        help='Quality requires depth and color to improve before trading point cost')
    parser.add_argument('--reuse-candidates',type=Path,
                        help='Reuse a report from this same candidate grid; rerun validation/selection')
    parser.add_argument('--min-overlap',type=float,default=.75,
                        help='Lower bound on the sigma/spacing ratio')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    configure_cuda()
    import torch
    torch.set_grad_enabled(False)
    K=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    cases=datasets();refs=[analytic(l,K,640,480) for _,l in cases]
    baseline_rows=[]
    for (name,layers),ref in zip(cases,refs):
        sampled=sample_layers(layers,.018,2.)
        # Historical density estimator is retained for the incumbent reference.
        item=[]
        for s in sampled:
            sc,q,_=covariance(s['p'],s['n'],s['edges'],s['h'],.8)
            item.append(dict(**s,sc=sc,q=q))
        r,rgb=render(item,.99,K,640,480,ref);baseline_rows.append(r)
        Image.fromarray((rgb*255).round().astype('uint8')).save(args.output_dir/(name+'_baseline.png'))
    base=aggregate(baseline_rows)
    rows=[];start=time.perf_counter()
    if args.reuse_candidates:
        cached=json.loads(args.reuse_candidates.read_text())
        if cached.get('support_model')!='directional-fixed-opacity-v1':
            raise ValueError('Cached grid predates the retained model; rerun the calibration grid')
        if not np.allclose([base[k] for k in base],[cached['baseline'][k] for k in base],rtol=1e-5,atol=1e-6):
            raise ValueError('Cached candidates have a different baseline; recalibrate the grid')
        rows=cached['candidates']
    for spacing,weight in (() if rows else itertools.product((.018,.022,.026),(2.,4.,8.))):
        sampled=[sample_layers(l,spacing,weight) for _,l in cases]
        for overlap in (.7,.75,.8):
            shaped=[shapes(s,overlap) for s in sampled]
            for opacity in (.99,.995):
                measurements=[render(s,opacity,K,640,480,ref)[0] for s,ref in zip(shaped,refs)]
                avg=aggregate(measurements)
                feasible=(min(r['coverage'] for r in measurements)>=.99 and min(r['mean_alpha'] for r in measurements)>=.98
                          and avg['leakage']<=base['leakage']+.005)
                rows.append(dict(base_spacing=spacing,texture_weight=weight,overlap=overlap,opacity=opacity,
                                 average=avg,measurements=measurements,feasible=feasible))
        print(f'Calibrated spacing={spacing}, texture_weight={weight}; {len(rows)} candidates; {time.perf_counter()-start:.1f}s',flush=True)
    for candidate in rows:
        avg=candidate['average'];measurements=candidate['measurements']
        candidate.pop('validation',None)
        candidate['feasible']=(min(r['coverage'] for r in measurements)>=.99 and min(r['mean_alpha'] for r in measurements)>=.98
                               and avg['leakage']<=base['leakage']+.005
                               and candidate['overlap']>=args.min_overlap)
        if args.objective=='quality':
            candidate['feasible'] &= avg['depth_mae']<=base['depth_mae'] and avg['psnr_db']>=base['psnr_db']
        cost_weight=.5 if args.objective=='quality' else 2.5
        candidate['utility']=float(avg['psnr_db']-cost_weight*np.log2(avg['points']/base['points'])-30*avg['depth_mae']-10*avg['leakage'])
    feasible=[r for r in rows if r['feasible']]
    if not feasible:raise RuntimeError('No calibration candidate meets coverage and leakage constraints')
    # Validate on different slopes and texture frequency. Reject candidates
    # whose anisotropic footprints leave holes in the new configuration.
    validation=datasets(held_out=True)
    validation_refs=[analytic(l,K,640,480) for _,l in validation]
    selected=None
    for candidate in sorted(feasible,key=lambda r:r['utility'],reverse=True):
        measured=[]
        for (name,layers),ref in zip(validation,validation_refs):
            shaped=shapes(sample_layers(layers,candidate['base_spacing'],candidate['texture_weight']),candidate['overlap'])
            r,_=render(shaped,candidate['opacity'],K,640,480,ref);measured.append(dict(case=name,**r))
        candidate['validation']=measured
        if min(r['coverage'] for r in measured)>=.99 and min(r['mean_alpha'] for r in measured)>=.98:
            selected=candidate;break
    if selected is None:raise RuntimeError('All feasible training candidates failed validation coverage')
    profile={k:selected[k] for k in ['base_spacing','texture_weight','overlap','opacity']}
    profile.update(curvature_weight=1.,detail_scale=1.,detail_weight=.25,max_normal_rotation=.22,curvature_power=2,thin_support=True,
                   directional_support=True)
    (args.output_dir/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    held=[]
    for name,layers in datasets(held_out=True):
        ref=analytic(layers,K,640,480)
        shaped=shapes(sample_layers(layers,profile['base_spacing'],profile['texture_weight']),profile['overlap'])
        r,rgb=render(shaped,profile['opacity'],K,640,480,ref)
        held.append(dict(case=name,**r));Image.fromarray((rgb*255).round().astype('uint8')).save(args.output_dir/(name+'_heldout.png'))
    report=dict(baseline=base,baseline_measurements=baseline_rows,selected=selected,profile=profile,held_out=held,
                candidates=rows,objective=args.objective,cached_candidates=str(args.reuse_candidates) if args.reuse_candidates else None,
                support_constraints=dict(min_overlap=args.min_overlap),
                selection=f'Coverage >=99%, mean alpha >=.98, bounded leakage; maximize PSNR -{cost_weight}log2(point cost)-30*depth MAE-10*leakage; quality also requires training depth/color no worse than baseline',
                support_model='directional-fixed-opacity-v1',no_scene_images_read=True,seconds=time.perf_counter()-start)
    (args.output_dir/'calibration_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(profile=profile,baseline=base,selected=selected['average'],held_out=held),indent=2),flush=True)


if __name__=='__main__':main()
