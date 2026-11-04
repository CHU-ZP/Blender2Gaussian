"""Repair a joint-optimized coverage model without a Gaussian count limit."""
import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import time

import numpy as np
from plyfile import PlyData

from mesh_to_gs.adaptive import TextureCache
from mesh_to_gs.color import srgb_to_linear
from mesh_to_gs.coverage import CoverageConfig, Sampler, array_digest, certify_model, digest, probe_model, read_model
from mesh_to_gs.coverage_repair import seed_sampler
if __package__:
    from .initialize_coverage_surface import combine
else:
    from initialize_coverage_surface import combine


def main():
    import bpy
    from mesh_to_gs.blend import extract_scene, load_texture
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('initializer', type=Path)
    p.add_argument('optimized', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    initial, learned, output = (x.resolve() for x in (args.initializer, args.optimized, args.output_dir))
    source_report = json.loads((initial / 'coverage_report.json').read_text())
    refinement = json.loads((learned / 'refinement_report.json').read_text())
    source = Path(source_report['input'])
    quality = CoverageConfig(**source_report['quality_protocol'])
    if source_report['status'] != 'complete' or digest(initial / 'model.ply') != source_report['model_sha256']:
        raise ValueError('Initializer certificate is incomplete or stale')
    if digest(source) != source_report['source_sha256'] or refinement['source_sha256'] != digest(source):
        raise ValueError('Original Mesh changed or optimized model uses a different source')
    if refinement['input_sha256'] != digest(initial / 'model.ply') or refinement['model_sha256'] != digest(learned / 'model.ply'):
        raise ValueError('Optimization lineage does not match the two actual PLYs')
    protocol = dict(source_sha256=digest(source), initializer_sha256=digest(initial / 'model.ply'),
                    optimized_sha256=digest(learned / 'model.ply'), lineage_sha256=digest(learned / 'lineage.npy'),
                    quality=asdict(quality), seed_opacity_ceiling=quality.opacity)
    if output.exists() and not args.resume:
        raise SystemExit('Use a new output directory or --resume; raw optimized models are preserved')
    output.mkdir(parents=True, exist_ok=True)
    # Serialize two resumed workers for the same output. Independent cases
    # can run concurrently without racing PLYs or pending target queues.
    repair_lock = (output / '.repair.lock').open('a')
    fcntl.flock(repair_lock, fcntl.LOCK_EX)
    checkpoint = output / 'checkpoint.json'
    manifest = json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(protocol=protocol, completed=[], status='incomplete')
    if manifest['protocol'] != protocol:
        raise ValueError('Repair checkpoint protocol changed')
    if args.resume and manifest['status'] == 'complete':
        complete = json.loads((output / 'coverage_report.json').read_text())
        saved = json.loads((output / 'repair_report.json').read_text())
        target = output / 'surface'
        target_hash = array_digest(*(np.load(target / f'{name}.npy', mmap_mode='r') for name in
                                    ('triangles', 'chart', 'face', 'certificate_face', 'certificate_barycentric')))
        if (complete['status'] != 'complete' or saved['status'] != 'complete' or
                complete['model_sha256'] != digest(output / 'model.ply') or
                complete['target_sha256'] != target_hash or saved['protocol'] != protocol):
            raise ValueError('Completed repair checkpoint changed; refusing stale certificate')
        print('Complete repair checkpoint and serialized target hashes verified:', output, flush=True)
        return
    if not checkpoint.exists() and list((output / 'groups').glob('*/pending.npz')):
        raise ValueError('Unbound learned-seed checkpoint: missing source/optimizer manifest')
    manifest.setdefault('started_at_unix', (output / '.repair.lock').stat().st_mtime)
    checkpoint.write_text(json.dumps(manifest, indent=2) + '\n')
    tick = time.perf_counter()
    points, scales, rotations, opacity = read_model(learned / 'model.ply')
    data = PlyData.read(learned / 'model.ply')['vertex'].data
    colours = srgb_to_linear(np.clip(np.column_stack([data[f'f_dc_{i}'] for i in range(3)]) * .28209479177387814 + .5, 0, 1))
    roots = np.load(learned / 'lineage.npy')
    initial_faces = np.load(initial / 'surface/face.npy')
    if len(roots) != len(points) or np.any((roots < 0) | (roots >= len(initial_faces))):
        raise ValueError('Invalid optimized split lineage')
    owning = initial_faces[roots]
    original_groups = json.loads((initial / 'surface/groups.json').read_text())
    bpy.ops.wm.open_mainfile(filepath=str(source.resolve()), load_ui=False)
    extracted, bounds, _ = extract_scene()
    source_groups = {(g['object_name'], g['description']['name']): g for g in extracted.values()}
    cache = TextureCache(load_texture)
    for index, original in enumerate(original_groups):
        directory = output / 'groups' / f'{index:06d}'
        if index < len(manifest['completed']):
            if digest(directory / 'model.ply') != manifest['completed'][index]['coverage']['model_sha256']:
                raise ValueError('Completed repaired group PLY changed')
            continue
        directory.mkdir(parents=True, exist_ok=True)
        group = source_groups[(original['object'], original['material'])]
        desc = group['description']
        texture = cache.get(desc['image'])
        uv = np.concatenate(group['uvs']) if group['uvs'] else None
        state = directory / 'pending.npz'
        sampler = Sampler(np.concatenate(group['triangles']), np.concatenate(group['vertex_ids']), uv,
                          None if texture is None else texture.image, desc['color'], quality,
                          state=state if state.exists() else None)
        original_group = source_report['groups'][index]['coverage']
        if (sampler.geometry_hash != original_group['geometry_sha256'] or
                sampler.material_hash != original_group['material_sha256']):
            raise ValueError('Extracted geometry/material differs from certified source')
        start, stop = original['faces']
        ids = np.flatnonzero((owning >= start) & (owning < stop))
        def progress(s):
            print(f'Group {index + 1}/{len(original_groups)}: samples={len(s.points):,}, '
                  f'certificates={len(s.cert_faces):,}, pending={len(s.pending):,}', flush=True)
        stats = seed_sampler(sampler, points[ids], scales[ids], rotations[ids], opacity[ids], colours[ids],
                             owning[ids] - start, checkpoint=state, progress=progress)
        sampler.run(checkpoint=state, progress=progress)
        report = sampler.export(directory)
        if report['status'] != 'complete':
            raise RuntimeError('Repaired export failed continuous coverage or quality certification')
        stats['added_deficit_samples'] = len(sampler.points) - len(ids)
        rgb_delta = np.abs(np.asarray(sampler.colours)[:len(ids)] - colours[ids])
        stats['maximum_learned_colour_adjustment_linear'] = float(rgb_delta.max()) if len(ids) else 0.
        stats['mean_learned_colour_adjustment_linear'] = float(rgb_delta.mean()) if len(ids) else 0.
        stats['opacity_preserved'] = bool(np.array_equal(np.asarray(sampler.opacities)[:len(ids)], opacity[ids]))
        stats['opacity_ceiling_applications'] = int((opacity[ids] > quality.opacity).sum())
        np.save(directory / 'surface/loop_triangle.npy', np.concatenate(group['loop_triangles'])[sampler.surface.source])
        np.save(directory / 'surface/optimized_row.npy', np.r_[ids, np.full(stats['added_deficit_samples'], -1, np.int64)])
        manifest['completed'].append(dict(directory=str(directory.relative_to(output)), object=original['object'],
            material=original['material'], colour=original['color'], has_uv=original['has_uv'],
            image=original['image'], coverage=report, repair=stats))
        checkpoint.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(stats), flush=True)
    combined = combine(output, manifest['completed'], protocol['source_sha256'],
                       SimpleNamespace(quality=quality, input=source, objects=None), bounds)
    combined.update(certify_model(output / 'model.ply', output / 'surface', quality))
    combined['training'] = 'Optimization before final uncapped repair and serialized-model recertification'
    combined['scope'] = source_report['scope']
    probes = probe_model(output / 'model.ply', output / 'surface', quality)
    combined['independent_probes'] = probes
    if probes['failed_probes'] or any(g['coverage']['quality']['status'] != 'complete' for g in manifest['completed']):
        combined['status'] = 'incomplete'
    for name in ('coverage_report.json', 'initialization_report.json'):
        (output / name).write_text(json.dumps(combined, indent=2) + '\n')
    optimized_rows = np.concatenate([np.load(output / g['directory'] / 'surface/optimized_row.npy') for g in manifest['completed']])
    np.save(output / 'surface/optimized_row.npy', optimized_rows)
    np.save(output / 'surface/repair_reason.npy',
            np.concatenate([np.load(output / g['directory'] / 'surface/repair_reason.npy') for g in manifest['completed']]))
    repair = dict(status=combined['status'], protocol=protocol, optimized_seeds=len(points),
                  final_gaussians=combined['gaussians'], added_deficit_samples=combined['gaussians'] - len(points),
                  gaussian_count_limit=None, groups=[g['repair'] for g in manifest['completed']],
                  opacity_preserved=all(g['repair']['opacity_preserved'] for g in manifest['completed']),
                  opacity_ceiling=quality.opacity,
                  opacity_ceiling_applications=sum(g['repair']['opacity_ceiling_applications'] for g in manifest['completed']),
                  seconds=time.perf_counter() - tick, raw_model=str(learned / 'model.ply'),
                  wall_seconds=time.time() - manifest['started_at_unix'],
                  seed_colour_rule='retain trained linear RGB within source centre +/- 0.5*colour_error; certify full footprint',
                  seed_opacity_rule='preserve trained opacity below initializer ceiling; cap near-one opacity to avoid exclusive early termination in gsplat 1.5.3',
                  covariance_rule='project learned covariance into source tangent plane; constrain spacing, boundary and colour',
                  stop_condition='every original source triangle continuously certified; no count/growth ceiling')
    (output / 'repair_report.json').write_text(json.dumps(repair, indent=2) + '\n')
    manifest['status'] = combined['status']
    checkpoint.write_text(json.dumps(manifest, indent=2) + '\n')
    root = Path(__file__).resolve().parents[1]
    for name in ('scripts/repair_coverage.py', 'scripts/initialize_coverage_surface.py',
                 'mesh_to_gs/coverage.py', 'mesh_to_gs/coverage_repair.py'):
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    if combined['status'] != 'complete':
        raise RuntimeError('Final repaired model did not pass recertification')
    print(json.dumps(repair), flush=True)


if __name__ == '__main__':
    main()
