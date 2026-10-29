"""Summarize quality, point count and actual CUDA raster time of the pipeline."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/material_pipeline'))
    p.add_argument('--reference',type=Path,default=Path('outputs/sk_00918_geometry_reference'))
    p.add_argument('--cameras',type=Path,default=Path('outputs/sk_00918_validation/renders/cameras.json'))
    p.add_argument('--final',type=Path,default=Path('outputs/sk_00918_stage3_directional'))
    p.add_argument('--final-render',default='renders_classic_005')
    p.add_argument('--final-evaluation',default='evaluation_final')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    definitions=[('Original BaseColor','sk_00918_surface_physical','renders','evaluation'),
                 ('Linear composition','sk_00918_stage1_linear','renders','evaluation'),
                 ('Screen filter calibration','sk_00918_stage1_linear','renders_classic_005','evaluation_classic_005'),
                 ('Adaptive sampling, original filter','sk_00918_stage2_adaptive','renders','evaluation'),
                 ('Adaptive sampling, calibrated filter','sk_00918_stage2_adaptive','renders_classic_005','evaluation_classic_005'),
                 ('Joint calibration, fast trial','sk_00918_stage3_calibrated','renders','evaluation'),
                 ('Texture reshaping, rejected','sk_00918_stage3_quality','renders','evaluation'),
                 ('Support without extension, rejected','sk_00918_stage3_supported','renders','evaluation'),
                 ('Directional support, original filter',str(args.final),'renders','evaluation'),
                 ('Joint calibration, selected',str(args.final),args.final_render,args.final_evaluation)]
    rows=[]
    for label,folder,render,evaluation in definitions:
        root=Path(folder) if '/' in folder else Path('outputs')/folder
        path=root/render
        rr=json.loads((path/'render_report.json').read_text())
        m=json.loads((root/evaluation/'metrics.json').read_text())['candidate_mean']
        init=root/'initialization_report.json'
        if not init.exists():init=Path(rr['source_ply']).parent/'initialization_report.json'
        ir=json.loads(init.read_text()) if init.exists() else None
        times=[v['cuda_raster_milliseconds'] for v in rr['views'] if 'cuda_raster_milliseconds' in v]
        rows.append(dict(stage=label,folder=str(root),renders=str(path),evaluation=str(root/evaluation),gaussians=rr['gaussians'],
                         rasterize_mode=rr['rasterize_mode'],eps2d=rr['eps2d'],
                         cuda_raster_ms_mean=float(np.mean(times)) if times else None,
                         cuda_raster_fps=float(1000/np.mean(times)) if times else None,
                         peak_cuda_gib=rr.get('total_peak_cuda_allocated_bytes',rr['peak_cuda_allocated_bytes'])/1024**3,
                         initialization_seconds=ir.get('total_seconds') if ir else None,
                         metrics=m))
    report=dict(stages=rows,reference=str(args.reference),scope='Full .blend, opaque unlit BaseColor, 1280x720 fixed cameras; variable point budgets',
                timing='CUDA events cover rasterization only; PNG/NPY output and PLY load excluded. CUDA allocation excludes driver/context memory.')
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    cameras=json.loads(args.cameras.read_text())['cameras']
    # Show stages that changed the material model; linear-only change is in the
    # numerical report because it is visually subtle at this thumbnail size.
    columns=[(args.reference,'Mesh BaseColor'),(Path('outputs/sk_00918_stage1_linear/renders_classic_005'),'Filter calibration'),
             (Path('outputs/sk_00918_stage2_adaptive/renders_classic_005'),'Adaptive sampling'),
             (args.final/args.final_render,'Joint calibration')]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    sheet=Image.new('RGB',(4*640,len(cameras)*390),'white');draw=ImageDraw.Draw(sheet)
    for row,c in enumerate(cameras):
        for col,(path,title) in enumerate(columns):
            image=Image.open(path/(c['name']+'.png')).convert('RGB').resize((640,360),Image.Resampling.LANCZOS)
            sheet.paste(image,(col*640,row*390));draw.text((col*640+8,row*390+365),c['name']+' | '+title,font=font,fill='black')
    sheet.save(args.output_dir/'all_stages.png')
    regions=[('Floor lettering','01_original_camera',(760,500,1000,710)),
             ('Book spines','04_seating_area',(980,440,1095,555)),
             ('Thin rods','04_seating_area',(325,350,400,425)),
             ('Curved foreground','02_lounge_corner',(65,350,560,630))]
    detail=Image.new('RGB',(4*480,len(regions)*340),'white');draw=ImageDraw.Draw(detail)
    detail_metrics=[]
    for row,(name,camera,box) in enumerate(regions):
        reference=np.asarray(Image.open(args.reference/(camera+'.png')).convert('RGB').crop(box),float)/255
        measurements=[]
        for col,(path,title) in enumerate(columns):
            crop=Image.open(path/(camera+'.png')).convert('RGB').crop(box)
            pixels=np.asarray(crop,float)/255;mse=np.mean((pixels-reference)**2)
            measurements.append(dict(stage=title,psnr_db=float(-10*np.log10(mse)) if mse>0 else None))
            preview=ImageOps.contain(crop,(480,300),Image.Resampling.NEAREST)
            detail.paste(preview,(col*480+(480-preview.width)//2,row*340+(300-preview.height)//2))
            draw.text((col*480+8,row*340+304),name+' | '+title,font=font,fill='black')
        detail_metrics.append(dict(name=name,camera=camera,pixel_box=box,stages=measurements))
    detail.save(args.output_dir/'detail_crops.png')
    report['detail_regions']=detail_metrics
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
