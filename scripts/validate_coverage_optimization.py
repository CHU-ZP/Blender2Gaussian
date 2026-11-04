"""Compare legacy/new initializers with matched joint optimization and final coverage repair."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw
from mesh_to_gs.coverage import CoverageConfig, certify_model, probe_model
from mesh_to_gs.mesh_supervision import digest

ROOT = Path(__file__).resolve().parents[1]
CASES = ['plane', 'hole', 'thin', 'checker', 'real_sk_00918', 'real_riliaodian']


def training_cameras(source, views):
    previous = json.loads((source/'cameras.json').read_text())
    report = json.loads((source/'coverage/coverage_report.json').read_text())
    center = np.asarray(report['source_bounds']).mean(0)
    span = np.linalg.norm(np.asarray(previous['cameras'][0]['camera_to_world'])[:3, 3]-center)/2.1
    basis = (np.asarray(json.loads((source/'source_selection.json').read_text())['camera_basis'])
             if (source/'source_selection.json').exists() else np.eye(3))
    entries = []
    for i in range(views):
        angle = 2*np.pi*i/views+.23
        side = 1 if i < (views+1)//2 else -1
        offset = np.array([.85*np.cos(angle), .65*np.sin(angle), side*(1.75+.22*np.sin(3*angle))])
        eye = center+basis@offset*span
        forward = center-eye; forward /= np.linalg.norm(forward)
        right = np.cross(forward, basis[:, 1]); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        matrix = np.eye(4); matrix[:3, :3] = np.column_stack((right, down, forward)); matrix[:3, 3] = eye
        for c in previous['cameras']:
            assert np.linalg.norm(eye-np.asarray(c['camera_to_world'])[:3, 3]) > 1e-6
        entries.append(dict(name=f'train_{i:03d}', role='train', camera_to_world=matrix.tolist(),
                            view_matrix=np.linalg.inv(matrix).tolist(), K=previous['cameras'][0]['K'],
                            near=.001, far=previous['cameras'][0]['far']))
    return dict(width=previous['width'], height=previous['height'], cameras=entries,
                independent_test_cameras=str(source/'cameras.json'))


def audit_raw(initializer, optimized, coverage_source, quality):
    """Keep source targets fixed while remapping split lineage into their charts."""
    target = optimized/'coverage_audit_surface'
    if not target.exists():
        shutil.copytree(coverage_source/'surface', target)
        old = initializer/('surface' if (initializer/'coverage_report.json').exists() else 'surface_metadata')
        roots = np.load(optimized/'lineage.npy')
        faces = np.load(old/'face.npy')[roots]
        if old.name != 'surface':
            old_groups = json.loads((old/'groups.json').read_text())
            new_groups = json.loads((target/'groups.json').read_text())
            old_loops, new_loops = np.load(old/'loop_triangle.npy'), np.load(target/'loop_triangle.npy')
            mapping = np.full(len(old_loops), -1, np.int64)
            for g in old_groups:
                match = next(n for n in new_groups if (n['object'], n['material']) == (g['object'], g['material']))
                a,b = match['faces']; lookup = {int(new_loops[j]):j for j in range(a,b)}
                a,b = g['faces']
                for j in range(a,b): mapping[j] = lookup[int(old_loops[j])]
            faces = mapping[faces]
        np.save(target/'face.npy', faces)
    certificate = certify_model(optimized/'model.ply', target, quality)
    probes = probe_model(optimized/'model.ply', target, quality)
    result = dict(certificate=certificate, probes=probes,
                  note='Failed conservative cells are unproven; failed independent probes show actual support deficits')
    (optimized/'coverage_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def sheet(directory, source, methods):
    names = [c['name'] for c in json.loads((source/'cameras.json').read_text())['cameras']]
    columns = [('Mesh', source/'reference')]+list(methods.items())
    w,h,label = 384,288,28
    result = Image.new('RGB',(w*len(columns),(h+label)*len(names)),'white')
    draw = ImageDraw.Draw(result)
    for row,name in enumerate(names):
        for col,(method,path) in enumerate(columns):
            result.paste(Image.open(path/f'{name}.png').convert('RGB').resize((w,h)),(col*w,row*(h+label)))
            draw.text((col*w+4,row*(h+label)+h+5),f'{name} | {method}',fill='black')
    result.save(directory/'comparison.png')


def gallery(output, results):
    cards=[]
    for name,r in results.items():
        means=r['metrics']['models']
        stats='; '.join(f'{k}: {v["mean"]["all"]["basecolor_psnr_db"]:.3f} dB' for k,v in means.items())
        cards.append(f'<article><h2>{name}</h2><p>{html.escape(stats)}</p><a href="{name}/comparison.png"><img src="{name}/comparison.png"></a><p><a href="{name}/metrics.json">Metrics</a></p></article>')
    (output/'gallery.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Coverage sampling + optimization</title><style>body{font:16px system-ui;margin:24px;background:#eee}article{background:white;padding:16px;margin:20px 0}img{width:100%;height:auto}</style><h1>Coverage sampling + matched optimization</h1><p>Columns: Mesh, legacy initializer, legacy + optimization, coverage initializer, coverage + optimization, coverage + optimization + certified repair. Three held-out cameras per case.</p>'+''.join(cards)+'</html>')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sampling-results',type=Path,default=ROOT/'outputs/coverage_validation_20261103')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--case',action='append',choices=CASES)
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--views',type=int,default=12)
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--gpu',default='1')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--train-only',action='store_true')
    p.add_argument('--summary-output',type=Path)
    args=p.parse_args()
    output=args.output_dir.resolve(); source_root=args.sampling_results.resolve()
    if output.exists() and not args.resume: raise SystemExit('Use a new output directory or --resume')
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(steps=args.steps,views=args.views,seed=args.seed,source=str(source_root),
                  optimizer='joint Mesh/render; fixed normals; trainable position/scales/rotation/RGB/opacity; uncapped eligible splits')
    manifest=output/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol: raise SystemExit('Resume protocol changed')
    manifest.write_text(json.dumps(protocol,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu)
    def run(script,*rest):
        cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,rest)]
        print('Running',' '.join(cmd),flush=True)
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    saved=output/'validation.json'
    results=json.loads(saved.read_text()) if saved.exists() else {}
    for name in args.case or CASES:
        source=source_root/name; directory=output/name; directory.mkdir(exist_ok=True)
        if name in results: continue
        camera_file=directory/'training_cameras.json'
        if not camera_file.exists(): camera_file.write_text(json.dumps(training_cameras(source,args.views),indent=2)+'\n')
        initial=directory/'legacy_input'
        if not (initial/'model.ply').exists():
            run('initialize_blend_surface.py',source/'source.blend','--output-dir',initial,'--base-spacing','.024','--no-visibility','--surface-metadata')
        if digest(initial/'model.ply')!=digest(source/'legacy/model.ply'): raise ValueError('Legacy initializer did not reproduce the saved comparison model')
        common=['--steps',args.steps,'--width','640','--height','480','--views',args.views,
                '--seed',args.seed,'--sampling-reference-steps','0','--query-count','16384',
                '--mesh-pool','8192','--mesh-batch','1024','--centre-batch','1024',
                '--retain-all-surfaces','--uncapped-splits','--surface-query-mode','area']
        for method,model in [('legacy',initial),('coverage',source/'coverage')]:
            refined=directory/f'{method}_optimized'
            if not (refined/'refinement_report.json').exists():
                extra=(['--training-cameras',camera_file] if method=='legacy' else ['--reference-cache',directory/'legacy_optimized'])
                run('refine_joint.py',model,'--output-dir',refined,*common,*extra)
        if args.train_only: continue
        quality=CoverageConfig(**json.loads((source/'coverage/coverage_report.json').read_text())['quality_protocol'])
        audits={method:audit_raw(model,directory/f'{method}_optimized',source/'coverage',quality)
                for method,model in [('legacy',initial),('coverage',source/'coverage')]}
        repaired=directory/'coverage_repaired'
        if not (repaired/'repair_report.json').exists() or json.loads((repaired/'repair_report.json').read_text())['status'] != 'complete':
            run('repair_coverage.py',source/'coverage',directory/'coverage_optimized','--output-dir',repaired,
                *(['--resume'] if repaired.exists() else []))
        methods={'legacy_init':source/'legacy_renders','legacy_opt':directory/'legacy_optimized_renders',
                 'coverage_init':source/'coverage_renders','coverage_opt':directory/'coverage_optimized_renders',
                 'coverage_repaired':directory/'coverage_repaired_renders'}
        for method,model in [('legacy_optimized',directory/'legacy_optimized'),('coverage_optimized',directory/'coverage_optimized'),('coverage_repaired',repaired)]:
            renders=directory/f'{method}_renders'
            if not (renders/'render_report.json').exists():
                run('render_gaussians.py',model/'model.ply','--output-dir',renders,'--cameras',source/'cameras.json','--geometry')
        params=[]
        for key,path in methods.items(): params+=['--model',key,path]
        run('compare_renders.py','--reference',source/'reference','--cameras',source/'cameras.json',*params,
            '--edge-tolerance-world','.001','--output',directory/'metrics.json')
        sheet(directory,source,methods)
        metrics=json.loads((directory/'metrics.json').read_text())
        views=metrics['models']['coverage_repaired']['views']
        acceptance=dict(full_foreground=all(v['full_foreground_opaque_coverage'] == 1. for v in views),
                        boundary=all(v['boundary_opaque_coverage'] == 1. for v in views),
                        outside_tails=all(v['outside_alpha_max_beyond_edge_band'] <= quality.outside_support_threshold for v in views),
                        continuous_certificate=json.loads((repaired/'coverage_report.json').read_text())['status'] == 'complete')
        acceptance['passed']=all(acceptance.values())
        results[name]=dict(metrics=metrics,audits=audits,acceptance=acceptance,
                           repair=json.loads((repaired/'repair_report.json').read_text()),
                           coverage=json.loads((repaired/'coverage_report.json').read_text()),
                           refinements={m:json.loads((directory/f'{m}_optimized/refinement_report.json').read_text()) for m in ['legacy','coverage']})
        saved.write_text(json.dumps(results,indent=2)+'\n');gallery(output,results)
    if args.summary_output and not args.train_only:
        args.summary_output.parent.mkdir(parents=True,exist_ok=True)
        summary=dict(protocol=protocol,artifacts=str(output),cases={name:dict(
            means={key:value['mean']['all'] for key,value in r['metrics']['models'].items()},
            raw_coverage={key:dict(certificate_status=a['certificate']['status'],minimum_lower_bound=a['certificate']['minimum_lower_bound'],failed_probes=a['probes']['failed_probes']) for key,a in r['audits'].items()},
            repair=r['repair'],certificate_status=r['coverage']['status'],acceptance=r['acceptance'],
            optimization={m:dict(gaussians=v['gaussians'],initial_count=v['initial_count'],optimization_seconds=v['optimization_seconds'],splits=v['splits'],gaussian_count_limit=v['gaussian_count_limit']) for m,v in r['refinements'].items()}) for name,r in results.items()})
        args.summary_output.write_text(json.dumps(summary,indent=2)+'\n')
    print('Optimization validation:',output,flush=True)


if __name__=='__main__': main()
