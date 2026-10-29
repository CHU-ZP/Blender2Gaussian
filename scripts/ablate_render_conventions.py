"""Analytic tilted planes, two-layer occlusion and black/white texture ablations.

Scene geometry is known exactly. No reference image is used for fitting.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion,binary_dilation
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from render_gaussians import configure_cuda
from mesh_to_gs.color import srgb_to_linear,linear_to_srgb
from mesh_to_gs.adaptive import bilinear


def plane(angle=0,center=(0,0,3),extent=(1,1),side=96,checker=0,color=(.5,.5,.5)):
    R=Rotation.from_euler('y',angle,degrees=True).as_matrix()
    gx,gy=np.meshgrid((np.arange(side)+.5)/side*2-1,(np.arange(side)+.5)/side*2-1)
    local=np.column_stack((gx.ravel()*extent[0],gy.ravel()*extent[1],np.zeros(gx.size)))
    means=local@R.T+center
    rgb=np.broadcast_to(color,(len(means),3)).copy()
    if checker:
        v=((np.floor((gx.ravel()+1)*checker/2)+np.floor((gy.ravel()+1)*checker/2))%2)
        rgb[:]=v[:,None]
    h=2*np.sqrt(extent[0]*extent[1])/side
    scales=np.tile([.8*h,.8*h,.016*h],(len(means),1))
    return dict(means=means,quats=np.tile(Rotation.from_matrix(R).as_quat(scalar_first=True),(len(means),1)),
                scales=scales,colors=rgb,R=R,center=np.array(center),extent=extent,checker=checker,color=color)


def analytic(layers,K,width,height,samples=4):
    rgb=np.zeros((height,width,3));alpha=np.zeros((height,width));depth=None
    yy,xx=np.mgrid[:height,:width]
    for sy in range(samples):
        for sx in range(samples):
            ray=np.stack(((xx+(sx+.5)/samples-K[0,2])/K[0,0],(yy+(sy+.5)/samples-K[1,2])/K[1,1],np.ones_like(xx)),axis=-1)
            closest=np.full((height,width),np.inf);color=np.ones((height,width,3))
            for p in layers:
                normal=p['R'][:,2]
                z=(normal@p['center'])/np.einsum('hwi,i->hw',ray,normal)
                local=(ray*z[...,None]-p['center'])@p['R']
                valid=(z>0)&(np.abs(local[...,0])<=p['extent'][0])&(np.abs(local[...,1])<=p['extent'][1])&(z<closest)
                c=np.broadcast_to(p['color'],color.shape).copy()
                if p['checker']:
                    value=(np.floor((local[...,0]/p['extent'][0]+1)*p['checker']/2)+np.floor((local[...,1]/p['extent'][1]+1)*p['checker']/2))%2
                    c[:]=value[...,None]
                feature=p.get('texture')
                if feature is not None:
                    uv=np.stack(((local[...,0]/p['extent'][0]+1)/2,(local[...,1]/p['extent'][1]+1)/2),axis=-1)
                    linear=bilinear(feature.image,uv.reshape(-1,2)).reshape(height,width,3)
                else:linear=srgb_to_linear(c)
                color[valid]=linear[valid];closest[valid]=z[valid]
            rgb+=color/(samples*samples);alpha+=np.isfinite(closest)/(samples*samples)
    # Center-ray first hit, separately from supersampled color/coverage.
    ray=np.stack(((xx+.5-K[0,2])/K[0,0],(yy+.5-K[1,2])/K[1,1],np.ones_like(xx)),axis=-1)
    depth=np.full((height,width),np.inf)
    for p in layers:
        normal=p['R'][:,2];z=(normal@p['center'])/np.einsum('hwi,i->hw',ray,normal)
        local=(ray*z[...,None]-p['center'])@p['R']
        valid=(z>0)&(np.abs(local[...,0])<=p['extent'][0])&(np.abs(local[...,1])<=p['extent'][1])
        depth=np.where(valid,np.minimum(depth,z),depth)
    return linear_to_srgb(rgb),alpha,np.where(np.isfinite(depth),depth,0)


def run_case(name,layers,K,width,height,output):
    import torch
    from gsplat import rasterization
    def cuda(x):return torch.as_tensor(np.asarray(x),dtype=torch.float32,device='cuda')
    kwargs={k:cuda(np.concatenate([p[k] for p in layers])) for k in ['means','quats','scales','colors']}
    kwargs.update(opacities=torch.full((len(kwargs['means']),),.99,device='cuda'),viewmats=cuda(np.eye(4)[None]),Ks=cuda(K[None]),width=width,height=height,packed=True,render_mode='RGB+ED',eps2d=.3)
    reference,refalpha,refdepth=analytic(layers,K,width,height)
    valid=binary_erosion(refalpha>.999,iterations=5)&(refdepth>0)
    rows=[];images=[reference]
    depths=[];alphas=[]
    for space in ('display','linear'):
        colors=kwargs['colors']
        render,alpha,_=rasterization(**{**kwargs,'colors':srgb_to_linear(colors) if space=='linear' else colors})
        color=render[0,:,:,:3]+1-alpha[0]
        if space=='linear':color=linear_to_srgb(color)
        rgb=color.cpu().numpy().clip(0,1);a=alpha[0,:,:,0].cpu().numpy();d=render[0,:,:,3].cpu().numpy()
        err=d-refdepth
        edges=np.zeros_like(valid)
        for axis in (0,1):edges|=np.abs(np.gradient(refdepth,axis=axis))>.2
        edge=binary_dilation(edges,iterations=2)&valid
        mse=np.mean((rgb[valid]-reference[valid])**2)
        rows.append(dict(space=space,psnr_db=float(-10*np.log10(max(mse,1e-12))),rgb_mae=float(np.abs(rgb-reference)[valid].mean()),
                         alpha_coverage=float((a[valid]>=.95).mean()),depth_mae=float(np.abs(err)[valid].mean()),depth_bias=float(err[valid].mean()),
                         edge_depth_mae=float(np.abs(err)[edge].mean()) if edge.any() else None))
        images.append(rgb);depths.append(d);alphas.append(a)
    assert np.array_equal(depths[0],depths[1]) and np.array_equal(alphas[0],alphas[1])
    sheet=Image.new('RGB',(width*3,height+30),'white');draw=ImageDraw.Draw(sheet)
    for i,(rgb,label) in enumerate(zip(images,('Analytic linear reference','Display-space composition','Linear-space composition'))):
        sheet.paste(Image.fromarray((rgb*255).round().astype('uint8')),(width*i,0));draw.text((width*i+8,height+8),label,fill='black')
    sheet.save(output/(name+'.png'))
    np.savez_compressed(output/(name+'.npz'),reference_depth=refdepth,gaussian_depth=depths[0],alpha=alphas[0],mask=valid)
    return dict(case=name,measurements=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=Path('outputs/render_conventions'));args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True);configure_cuda()
    import torch
    torch.set_grad_enabled(False)
    K=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    cases=[('plane_'+str(angle),[plane(angle)]) for angle in (0,45,75)]
    cases += [('occlusion',[plane(0,(0,0,3.5),extent=(1.5,1.2),color=(0,0,1)),plane(25,(-.35,0,2.5),extent=(.45,.7),side=64,color=(1,0,0))]),
              ('checker',[plane(checker=32)])]
    report=dict(reference='Analytic first hit and 4x4 supersampled scene-linear color',cases=[])
    for name,layers in cases:
        r=run_case(name,layers,K,640,480,args.output_dir);report['cases'].append(r);print(json.dumps(r),flush=True)
    report['analytic_half_black_half_white']=dict(display_average=.5,linear_average_srgb=float(linear_to_srgb(np.array(.5))))
    report['interpretation']='Color-space transfer is exact in the controlled mixture contract; spatial checker error also contains footprint blur and equal-depth alpha ordering. A linear pipeline need not improve MSE when those errors dominate.'
    report['depth_policy']='Keep expected camera-Z labelled explicitly; do not relabel it as first-hit depth'
    (args.output_dir/'ablation_report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
