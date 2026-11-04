"""Render matched Mesh/legacy/uncapped models in new experiment directories.

The fixtures include a boundary/corners, a hole, a thin strip and checker
texture. Optional isolated objects from both production assets are copied to
new .blend files; source assets and retained best models are never modified.
"""
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

ROOT = Path(__file__).resolve().parents[1]


def cameras(center, span, basis=None):
    width, height, focal = 640, 480, 560.
    basis = np.eye(3) if basis is None else np.asarray(basis)
    up = basis[:, 1]
    entries = []
    for name, offset in [('front', [0, 0, 2.1]), ('oblique', [1.2, -.8, 1.7]),
                         ('close', [.15, .1, 1.4])]:
        eye = np.asarray(center) + basis @ np.asarray(offset) * span
        forward = np.asarray(center) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        c2w = np.eye(4)
        c2w[:3, :3] = np.column_stack((right, down, forward))
        c2w[:3, 3] = eye
        entries.append(dict(name=name, camera_to_world=c2w.tolist(), view_matrix=np.linalg.inv(c2w).tolist(),
                            K=[[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                            near=.001, far=max(10., span * 10)))
    return dict(width=width, height=height, cameras=entries)


def fixture(directory, kind):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    a = .15
    if kind == 'hole':
        outer = np.array([[-a, -a, 0], [a, -a, 0], [a, a, 0], [-a, a, 0]])
        inner = outer * .3
        vertices = np.r_[outer, inner]
        faces = [[j, (j+1) % 4, (j+1) % 4 + 4, j + 4] for j in range(4)]
    elif kind == 'thin':
        vertices = np.array([[-a, -.0015, 0], [a, -.0015, 0], [a, .0015, 0], [-a, .0015, 0]])
        faces = [[0, 1, 2, 3]]
    else:
        vertices = np.array([[-a, -a, 0], [a, -a, 0], [a, a, 0], [-a, a, 0]])
        faces = [[0, 1, 2, 3]]
    mesh = bpy.data.meshes.new(kind)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(kind, mesh)
    bpy.context.collection.objects.link(obj)
    material = bpy.data.materials.new('BaseColor')
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (.08, .25, .6, 1)
    if kind == 'checker':
        checker = ((np.indices((16, 16))[0] // 4 + np.indices((16, 16))[1] // 4) % 2)
        pixels = np.where(checker[..., None], np.array([.8, .55, .25]), np.array([.2, .35, .65]))
        texture_path = directory / 'checker.png'
        Image.fromarray((pixels * 255).round().astype('uint8')).save(texture_path)
        image = bpy.data.images.load(str(texture_path.resolve()))
        image.pack()
        node = material.node_tree.nodes.new('ShaderNodeTexImage')
        node.image = image
        material.node_tree.links.new(node.outputs['Color'], bsdf.inputs['Base Color'])
        layer = mesh.uv_layers.new(name='UVMap')
        for loop in mesh.loops:
            xy = vertices[loop.vertex_index, :2]
            layer.data[loop.index].uv = ((xy + a) / (2 * a)).tolist()
    obj.data.materials.append(material)
    path = directory / 'source.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(path.resolve()))
    return path, cameras([0, 0, 0], .3)


def real_fixture(directory, source):
    """Choose a bounded-UV textured object for a close-up render experiment.

    This fixture selection is separate from the uncapped sampler. Hundreds
    of texture wraps on a small face are a different stress test; interrupted
    runs are preserved and cannot enter the completed results.
    """
    import bpy
    from mathutils import Vector
    from mesh_to_gs.blend import material_description
    bpy.ops.wm.open_mainfile(filepath=str(source.resolve()), load_ui=False)
    candidates = []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or obj.hide_render or not 6 <= len(obj.data.polygons) <= 300:
            continue
        world = np.array([obj.matrix_world @ Vector(v) for v in obj.bound_box])
        span = float(np.linalg.norm(world.max(0) - world.min(0)))
        if not .03 <= span <= .6:
            continue
        try:
            descriptions = [material_description(slot.material) for slot in obj.material_slots]
        except ValueError:
            continue
        if any(d['image'] is not None for d in descriptions):
            layer = obj.data.uv_layers.active
            if layer is None:
                continue
            uv = np.empty((len(obj.data.loops), 2), np.float32)
            layer.data.foreach_get('uv', uv.ravel())
            if not np.isfinite(uv).all() or np.ptp(uv, axis=0).max() > 2:
                continue
            candidates.append((len(obj.data.polygons), obj.name, obj, world, span, uv))
    if not candidates:
        raise ValueError(f'No suitable textured object in {source}')
    _, name, selected, world, span, uv = sorted(candidates, key=lambda x: (x[0], x[1]))[0]
    for obj in bpy.context.scene.objects:
        obj.hide_render = obj != selected
    path = directory / 'source.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(path.resolve()))
    xyz = np.array([v.co[:] for v in selected.data.vertices], dtype=float)
    matrix = np.asarray(selected.matrix_world, dtype=float)
    xyz = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    _, vectors = np.linalg.eigh((xyz - xyz.mean(0)).T @ (xyz - xyz.mean(0)))
    x, y = vectors[:, 2], vectors[:, 1]
    x *= 1 if x[np.abs(x).argmax()] >= 0 else -1
    y *= 1 if y[np.abs(y).argmax()] >= 0 else -1
    basis = np.column_stack((x, y, np.cross(x, y)))
    (directory / 'source_selection.json').write_text(json.dumps(dict(
        original=str(source), object=name, source_changed=False, scope='isolated real object',
        selection='visible textured object; 6..300 polygons; 0.03..0.6 m diagonal; UV span <= 2; polygon count/name order',
        uv_bounds=[uv.min(0).tolist(), uv.max(0).tolist()],
        camera_basis=basis.tolist(), camera_orientation='front faces the shortest principal axis'), indent=2) + '\n')
    center = (world.min(0) + world.max(0)) / 2
    return path, cameras(center, span, basis)


def sheet(directory, entries):
    thumb_width, thumb_height, label = 480, 360, 30
    canvas = Image.new('RGB', (thumb_width * 3, (thumb_height + label) * len(entries)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, name in enumerate(entries):
        for col, method in enumerate(('reference', 'legacy_renders', 'coverage_renders')):
            img = Image.open(directory / method / f'{name}.png').convert('RGB')
            x, y = col * thumb_width, row * (thumb_height + label)
            canvas.paste(img.resize((thumb_width, thumb_height)), (x, y))
            draw.text((x + 6, y + thumb_height + 6), f'{name} | {method}', fill='black')
    canvas.save(directory / 'comparison.png')


def gallery(output, results):
    cards = []
    percent = lambda value: f'{value:.2%}' if value is not None else '无前景'
    for kind, result in results.items():
        old = result['metrics']['models']['legacy']['mean']['all']
        new = result['metrics']['models']['coverage']['mean']['all']
        outside = max((v['outside_alpha_max_beyond_edge_band'] or 0)
                      for v in result['metrics']['models']['coverage']['views'])
        accepted = all(result['render_acceptance'].values())
        cards.append(f'<article><h2>{html.escape(kind)}</h2><p>完整前景覆盖：'
                     f'{percent(old["full_foreground_opaque_coverage"])} → {percent(new["full_foreground_opaque_coverage"])}；'
                     f'高斯数量：{result["coverage"]["gaussians"]:,}；'
                     f'连续认证下界：{result["coverage"]["minimum_lower_bound"]:.6f}；'
                     f'轮廓带外最大 alpha：{outside:.6f}；验收：{"通过" if accepted else "未通过"}</p>'
                     f'<a href="{kind}/comparison.png"><img src="{kind}/comparison.png"></a>'
                     f'<p><a href="{kind}/metrics.json">逐视角指标</a> · '
                     f'<a href="{kind}/coverage/coverage_report.json">覆盖证书</a></p></article>')
    for report in sorted(output.glob('interrupted_*/interruption_report.json')):
        cards.append(f'<article><h2>未完成的压力测试</h2><p>高重复 UV 构件的运行已中断，'
                     f'保留完整检查点，未计入通过结果。</p><a href="{report.relative_to(output)}">中断报告</a></article>')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>覆盖采样渲染验证</title>'
            '<style>body{font:16px system-ui;max-width:1440px;margin:24px auto;padding:0 20px;'
            'background:#f4f5f7;color:#222}article{background:white;padding:20px;margin:24px 0}'
            'img{width:100%;height:auto}a{color:#1664ac}h1,h2{font-weight:600}</style>'
            '<h1>覆盖采样渲染验证</h1><p>每行从左到右：Mesh 参照、旧初始化、新覆盖采样。'
            '同相机、无光照、线性白背景合成。几何认证与屏幕 alpha 分别统计。'
            '真实场景验证使用独立复制的实际物体，未重训完整场景。</p>' + ''.join(cards) + '</html>')
    (output / 'gallery.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--skip-real', action='store_true')
    parser.add_argument('--case', action='append', choices=('plane', 'hole', 'thin', 'checker', 'real_sk_00918', 'real_riliaodian'),
                        help='Run selected cases and preserve other saved completed results')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report-only', action='store_true', help='Rebuild summaries/gallery from saved completed results without sampling or rendering')
    parser.add_argument('--summary-output', type=Path, help='Also save the compact metrics/protocol report here')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise SystemExit('Use a new validation directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    def run(script, *rest):
        command = [sys.executable, str(ROOT / 'scripts' / script), *map(str, rest)]
        print('Running', ' '.join(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT, env=env)
    cases = ['plane', 'hole', 'thin', 'checker']
    if not args.skip_real:
        cases += ['real_sk_00918', 'real_riliaodian']
    if args.case:
        cases = list(dict.fromkeys(args.case))
    saved = output / 'validation.json'
    results = json.loads(saved.read_text()) if (args.resume or args.report_only) and saved.exists() else {}
    if args.report_only and set(cases) - set(results):
        raise SystemExit('Saved validation does not contain every requested case')
    for kind in ([] if args.report_only else cases):
        directory = output / kind
        directory.mkdir(exist_ok=True)
        camera_path = directory / 'cameras.json'
        if not (directory / 'source.blend').exists():
            source, cam = (real_fixture(directory, ROOT / 'data' / (kind[5:] + '.blend'))
                           if kind.startswith('real_') else fixture(directory, kind))
            camera_path.write_text(json.dumps(cam, indent=2) + '\n')
        else:
            source = directory / 'source.blend'
            cam = json.loads(camera_path.read_text())
        if not (directory / 'legacy' / 'model.ply').exists():
            run('initialize_blend_surface.py', source, '--output-dir', directory / 'legacy',
                '--base-spacing', '.024', '--no-visibility')
        if not (directory / 'coverage' / 'model.ply').exists():
            # .08 is an explicit colour quality protocol for rendering
            # experiments. Production default remains .02 and is uncapped.
            run('initialize_coverage_surface.py', source, '--output-dir', directory / 'coverage',
                '--colour-error', '.08', *(['--resume'] if (directory / 'coverage').exists() else []))
        elif args.resume:
            run('initialize_coverage_surface.py', source, '--output-dir', directory / 'coverage',
                '--colour-error', '.08', '--resume', '--revalidate')
        if not (directory / 'reference' / 'reference_report.json').exists():
            run('render_geometry_reference.py', '--blend', source, '--cameras', camera_path,
                '--output-dir', directory / 'reference')
        for method in ('legacy', 'coverage'):
            renders = directory / f'{method}_renders'
            if not (renders / 'render_report.json').exists():
                run('render_gaussians.py', directory / method / 'model.ply', '--output-dir', renders,
                    '--cameras', camera_path, '--geometry')
        run('compare_renders.py', '--reference', directory / 'reference', '--cameras', camera_path,
            '--model', 'legacy', directory / 'legacy_renders', '--model', 'coverage', directory / 'coverage_renders',
            '--output', directory / 'metrics.json', '--edge-tolerance-world', '.001')
        sheet(directory, [c['name'] for c in cam['cameras']])
        from mesh_to_gs.coverage import CoverageConfig, probe_model
        protocol = json.loads((directory / 'coverage' / 'coverage_report.json').read_text())['quality_protocol']
        probes = probe_model(directory / 'coverage' / 'model.ply', directory / 'coverage' / 'surface',
                             CoverageConfig(**protocol))
        (directory / 'coverage' / 'probe_report.json').write_text(json.dumps(probes, indent=2) + '\n')
        results[kind] = dict(metrics=json.loads((directory / 'metrics.json').read_text()),
                             coverage=json.loads((directory / 'coverage' / 'coverage_report.json').read_text()),
                             probes=probes,
                             comparison=str(directory / 'comparison.png'))
        if (directory / 'source_selection.json').exists():
            results[kind]['source_selection'] = json.loads((directory / 'source_selection.json').read_text())
        rendered = results[kind]['metrics']['models']['coverage']['views']
        results[kind]['render_acceptance'] = dict(
            full_foreground_coverage_at_least_999=all(v['full_foreground_opaque_coverage'] is not None
                                                     and v['full_foreground_opaque_coverage'] >= .999 for v in rendered),
            boundary_coverage_at_least_999=all(v['boundary_opaque_coverage'] is None
                                              or v['boundary_opaque_coverage'] >= .999 for v in rendered),
            outside_alpha_within_005=all(v['outside_alpha_max_beyond_edge_band'] is None
                                         or v['outside_alpha_max_beyond_edge_band'] <= .05 for v in rendered),
            geometry_certificate_complete=results[kind]['coverage']['status'] == 'complete',
            independent_probes_pass=probes['failed_probes'] == 0)
        (output / 'validation.json').write_text(json.dumps(results, indent=2) + '\n')
        gallery(output, results)
    if args.report_only:
        results = {kind: results[kind] for kind in cases}
        for kind, result in results.items():
            views = result['metrics']['models']['coverage']['views']
            result['render_acceptance']['boundary_coverage_at_least_999'] = all(
                v['boundary_opaque_coverage'] is None or v['boundary_opaque_coverage'] >= .999 for v in views)
            selection = output / kind / 'source_selection.json'
            if selection.exists():
                result['source_selection'] = json.loads(selection.read_text())
        (output / 'validation.json').write_text(json.dumps(results, indent=2) + '\n')
        gallery(output, results)
    print('Validation artifacts:', output, flush=True)
    if args.summary_output:
        interrupted = {p.parent.name: json.loads(p.read_text())
                       for p in sorted(output.glob('interrupted_*/interruption_report.json'))}
        summary = dict(date='2026-11-04', artifacts=str(output),
                       scope='four controlled fixtures' + ('' if args.skip_real else ' and isolated bounded-UV textured objects from both source scenes'),
                       comparison='initializers; retained trained best models preserved',
                       renderer_protocol=dict(width=640, height=480, cameras_per_case=3, gpu='A100',
                                              colour_error_linear=.08, production_default_colour_error_linear=.02,
                                              background='white; linear compositing', lighting='unlit BaseColor',
                                              gsplat_eps2d=.05, rasterize_mode='classic', foreground_alpha_threshold=.95),
                       all_render_acceptance_pass=all(all(r['render_acceptance'].values()) for r in results.values()),
                       incomplete_stress_tests=interrupted,
                       cases={kind: dict(quality_protocol=r['coverage']['quality_protocol'],
                                         source_selection=r.get('source_selection'),
                                         gaussians=r['coverage']['gaussians'],
                                         legacy_gaussians=json.loads((output / kind / 'legacy_renders' / 'render_report.json').read_text())['gaussians'],
                                         initialization_seconds=(sum(g['coverage']['initialization_seconds'] for g in r['coverage']['groups'])
                                                                 if all(g['coverage'].get('initialization_seconds') is not None for g in r['coverage']['groups']) else None),
                                         construction_boundary_radius=r['coverage']['groups'][0]['coverage'].get('construction_boundary_radius', 3.),
                                         model_sha256=r['coverage']['model_sha256'],
                                         target_sha256=r['coverage']['target_sha256'],
                                         certificate_status=r['coverage']['status'],
                                         certificate_cells=r['coverage']['certificate_cells'],
                                         minimum_lower_bound=r['coverage']['minimum_lower_bound'],
                                         probes=r['probes'], render_acceptance=r['render_acceptance'],
                                         means={k: m['mean']['all'] for k, m in r['metrics']['models'].items()})
                              for kind, r in results.items()})
        if (output / 'preservation_report.json').exists():
            summary['preservation'] = json.loads((output / 'preservation_report.json').read_text())
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
