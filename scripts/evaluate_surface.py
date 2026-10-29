"""Compare unlit Gaussian geometry against matched opaque Mesh references."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from PIL import Image, ImageDraw, ImageFont


def contours(depth, valid):
    edge=np.zeros(depth.shape,bool)
    for axis in (0,1):
        first=[slice(None)]*2;second=[slice(None)]*2
        first[axis]=slice(None,-1);second[axis]=slice(1,None)
        a,b=tuple(first),tuple(second)
        jump=np.abs(depth[a]-depth[b])>np.maximum(.05,.02*np.minimum(depth[a],depth[b]))
        jump=(jump & valid[a] & valid[b]) | (valid[a] != valid[b])
        edge[a] |= jump;edge[b] |= jump
    return edge


def contour_score(reference, predicted):
    tolerance=2
    precision=np.sum(predicted & binary_dilation(reference,iterations=tolerance))/max(predicted.sum(),1)
    recall=np.sum(reference & binary_dilation(predicted,iterations=tolerance))/max(reference.sum(),1)
    return dict(precision=float(precision),recall=float(recall),f1=float(2*precision*recall/max(precision+recall,1e-12)))


def compare(reference, candidate, name):
    md=np.load(reference/(name+'_depth.npy'))
    ma=np.load(reference/(name+'_alpha.npy'))
    gd=np.load(candidate/(name+'_depth.npy'))
    ga=np.load(candidate/(name+'_alpha.npy'))
    mesh=(ma>=.99)&(md>0)
    me=contours(md,mesh)
    interior=binary_erosion(mesh,iterations=3)&~binary_dilation(me,iterations=3)
    opaque=ga>=.95
    ge=contours(gd,opaque&(gd>0))
    error=np.abs(gd-md)
    rgbm=np.asarray(Image.open(reference/(name+'.png')).convert('RGB'),float)/255
    rgbg=np.asarray(Image.open(candidate/(name+'.png')).convert('RGB'),float)/255
    mse=np.mean((rgbm[mesh]-rgbg[mesh])**2)
    metric=dict(camera=name, interior_pixels=int(interior.sum()),
                opaque_coverage=float(opaque[interior].mean()),
                depth_consistent_coverage=float((opaque & (error<=np.maximum(.03,.01*md)))[interior].mean()),
                depth_mae_world_units=float(error[interior].mean()),
                depth_p95_world_units=float(np.quantile(error[interior],.95)),
                depth_within_3cm=float((error[interior]<=.03).mean()),
                basecolor_psnr_db=float(-10*np.log10(max(mse,1e-12))),
                basecolor_mae=float(np.abs(rgbm-rgbg)[mesh].mean()),
                contour=contour_score(me,ge))
    # Diagnostic: red = missing alpha, orange = incorrect depth, blue = correct.
    debug=np.full((*md.shape,3),255,np.uint8)
    debug[mesh & opaque & (error<=np.maximum(.03,.01*md))]=[50,135,210]
    debug[mesh & opaque & (error>np.maximum(.03,.01*md))]=[244,170,45]
    debug[mesh & ~opaque]=[210,45,45]
    return metric,Image.fromarray(debug)


def geometry_figures(reference,candidate,cameras,diagnostics,metrics,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=[]
    for c in cameras:
        name=c['name']
        md=np.load(reference/(name+'_depth.npy'))
        gd=np.load(candidate/(name+'_depth.npy'))
        data.append((md,gd))
    limit=max(float(np.quantile(md[md>0],.995)) for md,_ in data)
    fig,axes=plt.subplots(len(cameras),3,figsize=(15,3*len(cameras)),layout='constrained')
    for row,(c,(md,gd)) in enumerate(zip(cameras,data)):
        for col,depth in enumerate((md,gd)):
            im=axes[row,col].imshow(np.where(md>0,depth,np.nan),vmin=0,vmax=limit,cmap='viridis')
        err=axes[row,2].imshow(np.where(md>0,np.abs(md-gd),np.nan),vmin=0,vmax=.15,cmap='magma')
        for col,title in enumerate(('Mesh first-hit Z','Gaussian expected Z','Absolute depth error')):
            axes[row,col].set_title(c['name']+' | '+title,fontsize=9)
            axes[row,col].axis('off')
    fig.colorbar(im,ax=axes[:,:2],label='Camera Z [m]',shrink=.7)
    fig.colorbar(err,ax=axes[:,2],label='Absolute error [m], clipped at 0.15',shrink=.7)
    fig.savefig(output/'depth_comparison.png',dpi=140)
    plt.close(fig)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    sheet=Image.new('RGB',(1600,40+486*2),'white');draw=ImageDraw.Draw(sheet)
    draw.text((10,10),'Blue: alpha >= .95 + correct depth | Orange: incorrect depth | Red: alpha < .95',font=font,fill='black')
    for i,(c,image,m) in enumerate(zip(cameras,diagnostics,metrics)):
        x,y=(i%2)*800,40+(i//2)*486
        sheet.paste(image.resize((800,450),Image.Resampling.NEAREST),(x,y))
        label=f"{c['name']} | opaque {m['opaque_coverage']:.1%} | depth-consistent {m['depth_consistent_coverage']:.1%}"
        draw.text((x+10,y+458),label,font=font,fill='black')
    sheet.save(output/'coverage_overview.png')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('outputs/sk_00918_geometry_reference'))
    p.add_argument('--baseline',type=Path,default=Path('outputs/sk_00918_scale_fixes/geometry_renders'))
    p.add_argument('--candidate',type=Path,default=Path('outputs/sk_00918_surface_physical/renders'))
    p.add_argument('--cameras',type=Path,default=Path('outputs/sk_00918_validation/renders/cameras.json'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/sk_00918_surface_physical/evaluation'))
    p.add_argument('--baseline-label',default='Baseline Gaussian')
    p.add_argument('--candidate-label',default='Candidate Gaussian')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    cameras=json.loads(args.cameras.read_text())['cameras']
    results={}
    diagnostics={}
    for label,path in [('baseline',args.baseline),('candidate',args.candidate)]:
        results[label]=[];diagnostics[label]=[]
        for c in cameras:
            metric,image=compare(args.reference,path,c['name'])
            results[label].append(metric);diagnostics[label].append(image)
            image.save(args.output_dir/(label+'_'+c['name']+'_coverage.png'))
        keys=['opaque_coverage','depth_consistent_coverage','depth_mae_world_units','depth_p95_world_units',
              'depth_within_3cm','basecolor_psnr_db','basecolor_mae']
        results[label+'_mean']={key:float(np.mean([m[key] for m in results[label]])) for key in keys}
        results[label+'_mean']['contour_f1']=float(np.mean([m['contour']['f1'] for m in results[label]]))
    results['definitions']={
        'interior':'Mesh opaque pixels eroded 3px, excluding 3px around Mesh depth contours',
        'opaque_coverage':'Gaussian alpha >= 0.95 on Mesh interior; alone does not guarantee correct geometry',
        'depth_consistent_coverage':'alpha >= 0.95 and camera-Z error <= max(3cm,1% Mesh depth) on interior',
        'depth':'Mesh first hit vs Gaussian alpha-normalized expected depth, world units; not identical surface estimators',
        'contour':'Depth jump > max(5cm,2% nearer depth) or mask boundary, F1 within 2px; misses smooth geometric/material edges',
        'psnr':'sRGB unlit BaseColor, all Mesh foreground pixels, no exposure matching',
        'scope':'Four fixed interior cameras; opaque BaseColor only; no lighting, no image-based fitting',
    }
    results['paths']={k:str(getattr(args,k)) for k in ['reference','baseline','candidate','cameras']}
    results['render_policies']={}
    for label,path in [('baseline',args.baseline),('candidate',args.candidate)]:
        file=path/'render_report.json'
        results['render_policies'][label]=json.loads(file.read_text()) if file.exists() else None
    (args.output_dir/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    sheet=Image.new('RGB',(800*3,486*len(cameras)),'white');draw=ImageDraw.Draw(sheet)
    for row,c in enumerate(cameras):
        for col,(path,label) in enumerate([(args.reference,'Mesh BaseColor'),(args.baseline,args.baseline_label),(args.candidate,args.candidate_label)]):
            image=Image.open(path/(c['name']+'.png')).convert('RGB').resize((800,450),Image.Resampling.LANCZOS)
            sheet.paste(image,(col*800,row*486))
            draw.text((col*800+10,row*486+458),c['name']+' | '+label,font=font,fill='black')
    sheet.save(args.output_dir/'mesh_baseline_surface.png')
    geometry_figures(args.reference,args.candidate,cameras,diagnostics['candidate'],results['candidate'],args.output_dir)
    print(json.dumps({k:v for k,v in results.items() if k.endswith('_mean')},indent=2))


if __name__=='__main__':main()
