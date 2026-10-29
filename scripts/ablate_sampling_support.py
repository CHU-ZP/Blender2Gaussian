"""Compare legacy kNN and directional support on an uneven sample layout."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from mesh_to_gs.surface import covariance
from calibrate_surface import layer,render
from ablate_render_conventions import analytic
from render_gaussians import configure_cuda


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/sampling_support'))
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    configure_cuda()
    import torch
    torch.set_grad_enabled(False)
    columns=np.arange(-15,16)
    # Alternating column gaps mimic locally uneven adaptive allocations.
    x,y=np.meshgrid(columns*.1+.025*np.where(columns%2,1.,-1.),np.arange(-150,151)*.01)
    p=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,3.)))
    n=np.tile([0.,0,1],(len(p),1))
    K=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    reference=analytic([layer(extent=(1.4,1.2))],K,640,480)
    measurements=[];sheet=Image.new('RGB',(1280,510),'white');draw=ImageDraw.Draw(sheet)
    for i,extended in enumerate((False,True)):
        sc,q,_,area=covariance(p,n,np.empty((0,2,3)),np.sqrt(.001),overlap=.7,
            directional_support=extended,return_areas=True)
        result,rgb=render([dict(p=p,n=n,sc=sc,q=q,colors=np.full((len(p),3),.5))],.995,K,640,480,reference)
        measurements.append(dict(directional_support=extended,coverage=result['coverage'],mean_alpha=result['mean_alpha']))
        sheet.paste(Image.fromarray((rgb.clip(0,1)*255).round().astype('uint8')),(i*640,0))
        draw.text((i*640+8,488),f"Directional support {extended} | coverage {result['coverage']:.2%}",fill='black')
    report=dict(layout='Surface columns with alternating 0.05/0.15m x gaps and 0.01m y spacing',
                measurements=measurements,interpretation='Directional neighbor extension and wider support bounds address sparse directions.')
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    sheet.save(args.output_dir/'support_comparison.png');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
