"""Build a certified surface model; quality determines count, without a budget.

All nondegenerate source geometry is retained. Unproved hidden layers remain
targets. Completed object/material groups are disk checkpoints. The current
group also saves its full pending queue and can resume after interruption.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np

from mesh_to_gs.adaptive import TextureCache
from mesh_to_gs.coverage import CoverageConfig, Sampler, array_digest, certify_model, digest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--revalidate', action='store_true', help='With --resume, recheck completed PLYs under the current implementation')
    parser.add_argument('--objects', help='Explicit object-name regex for an isolated experiment; default is the entire scene')
    for name, default in asdict(CoverageConfig()).items():
        parser.add_argument('--' + name.replace('_', '-'), type=type(default), default=default)
    args = parser.parse_args(argv)
    try:
        args.quality = CoverageConfig(**{k: getattr(args, k) for k in asdict(CoverageConfig())})
    except ValueError as error:
        parser.error(str(error))
    return args


def combine(output, completed, source_hash, args, bounds):
    """Stream PLY bodies and geometry/target caches; do not load all samples."""
    from plyfile import PlyData
    from numpy.lib.format import open_memmap
    counts, face_counts = [], []
    for group in completed:
        counts.append(group['coverage']['gaussians'])
        face_counts.append(group['coverage']['target_faces'])
    attributes = PlyData.read(output / completed[0]['directory'] / 'model.ply')['vertex'].data.dtype.names
    total = sum(counts)
    if not sum(face_counts):
        raise SystemExit('No nonzero-area surface remains; checkpoint is incomplete')
    total_certificates = sum(g['coverage']['certificate_cells'] for g in completed)
    header = 'ply\nformat binary_little_endian 1.0\nelement vertex ' + str(total) + '\n'
    header += ''.join(f'property double {a}\n' for a in attributes) + 'end_header\n'
    with (output / 'model.ply').open('wb') as stream:
        stream.write(header.encode())
        for group in completed:
            path = output / group['directory'] / 'model.ply'
            with path.open('rb') as body:
                while body.readline().strip() != b'end_header':
                    pass
                shutil.copyfileobj(body, stream, length=1 << 20)
    surface = output / 'surface'
    surface.mkdir(exist_ok=True)
    specs = dict(triangles=('f8', (sum(face_counts), 3, 3)), uv=('f8', (sum(face_counts), 3, 2)),
                 chart=('i8', (sum(face_counts),)), source=('i8', (sum(face_counts),)),
                 loop_triangle=('i8', (sum(face_counts),)),
                 face=('i8', (total,)), barycentric=('f8', (total, 2)), boundary=('?', (total,)),
                 sample=('i8', (total,)), target_face=('i8', (sum(face_counts),)),
                 target_state=('u1', (sum(face_counts),)),
                 certificate_face=('i8', (total_certificates,)),
                 certificate_barycentric=('f8', (total_certificates, 3, 3)),
                 certificate_lower_bound=('f8', (total_certificates,)))
    arrays = {key: open_memmap(surface / f'{key}.npy', mode='w+', dtype=dtype, shape=shape)
              for key, (dtype, shape) in specs.items()}
    face_offset = row_offset = chart_offset = certificate_offset = 0
    groups = []
    for g, nf, nr in zip(completed, face_counts, counts):
        directory = output / g['directory'] / 'surface'
        charts = np.load(directory / 'chart.npy', mmap_mode='r')
        for key in specs:
            value = np.load(directory / f'{key}.npy', mmap_mode='r')
            if key in ('face', 'target_face', 'certificate_face'):
                value = value + face_offset
            elif key == 'chart':
                value = value + chart_offset
            elif key == 'sample':
                value = value + row_offset
            is_row = key in ('face', 'barycentric', 'boundary', 'sample')
            if key.startswith('certificate_'):
                start, count = certificate_offset, g['coverage']['certificate_cells']
            else:
                start, count = (row_offset, nr) if is_row else (face_offset, nf)
            arrays[key][start:start + count] = value
        groups.append(dict(object=g['object'], material=g['material'], color=g['colour'],
                           has_uv=g['has_uv'], image=g['image'],
                           faces=[face_offset, face_offset + nf],
                           charts=[chart_offset, chart_offset + (int(charts.max()) + 1 if nf else 0)]))
        face_offset += nf
        row_offset += nr
        chart_offset = groups[-1]['charts'][1]
        certificate_offset += g['coverage']['certificate_cells']
    for array in arrays.values():
        array.flush()
    (surface / 'groups.json').write_text(json.dumps(groups, indent=2) + '\n')
    reports = [g['coverage'] for g in completed]
    all_complete = all(r['status'] == 'complete' for r in reports)
    report = dict(status='complete' if all_complete else 'incomplete',
                  model_sha256=digest(output / 'model.ply'), source_sha256=source_hash,
                  target_sha256=array_digest(arrays['triangles'], arrays['chart'], arrays['face'],
                                             arrays['certificate_face'], arrays['certificate_barycentric']),
                  quality_protocol=asdict(args.quality), gaussians=total,
                  target_faces=sum(face_counts), target_area=sum(r['target_area'] for r in reports),
                  certified_area=sum(r['certified_area'] for r in reports),
                  minimum_lower_bound=min(r['minimum_lower_bound'] for r in reports if r['minimum_lower_bound'] is not None),
                  unresolved_cells=sum(r['unresolved_cells'] for r in reports),
                  certificate_cells=sum(r['certificate_cells'] for r in reports),
                  count_stop_condition=False, source_bounds=bounds, input=str(args.input.resolve()),
                  scope='entire scene' if args.objects is None else f'object regex: {args.objects}',
                  groups=completed, certificate_stage='per-group serialized float64 PLY; combined body copied byte-for-byte',
                  training='No optimization after certification; modifying parameters invalidates certificates')
    (output / 'coverage_report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'initialization_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    import bpy
    from mesh_to_gs.blend import extract_scene, load_texture
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise SystemExit('Use a new output directory or --resume; existing models are preserved')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'checkpoint.json'
    source_hash = digest(args.input)
    protocol = dict(source_sha256=source_hash, quality=asdict(args.quality), objects=args.objects)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['protocol'] != protocol:
            raise SystemExit('Resume source or quality protocol changed')
    else:
        manifest = dict(protocol=protocol, status='incomplete', completed=[])
    manifest['status'] = 'incomplete'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()), load_ui=False)
    groups, bounds, objects = extract_scene()
    cache = TextureCache(load_texture)
    selected = [(key, g) for key, g in groups.items()
                if args.objects is None or re.search(args.objects, g['object_name'])]
    if not selected:
        raise SystemExit('No objects matched')
    visibility = dict(protocol='conservative entire evaluated surface', viewpoints=[],
                      hidden_regions_removed=0, reason='No whole-region occlusion proof; all uncertain layers retained',
                      objects=[g['object_name'] for _, g in selected])
    (output / 'visibility_report.json').write_text(json.dumps(visibility, indent=2) + '\n')
    for index, (_, group) in enumerate(selected):
        directory = output / 'groups' / f'{index:06d}'
        if index < len(manifest['completed']):
            previous = manifest['completed'][index]
            if digest(directory / 'model.ply') != previous['coverage']['model_sha256']:
                raise SystemExit('Completed checkpoint PLY changed; refusing stale certificate')
            if not args.revalidate:
                continue
        directory.mkdir(parents=True, exist_ok=True)
        state = directory / 'pending.npz'
        desc = group['description']
        texture = cache.get(desc['image'])
        triangles = np.concatenate(group['triangles'])
        uv = np.concatenate(group['uvs']) if group['uvs'] else None
        sampler = Sampler(triangles, np.concatenate(group['vertex_ids']), uv,
                          image=None if texture is None else texture.image, colour=desc['color'],
                          config=args.quality, state=state if state.exists() else None)
        print(f'Group {index + 1}/{len(selected)}: {group["object_name"]} / {desc["name"]}; '
              f'{len(sampler.surface.areas):,} independent target faces', flush=True)
        tick = time.perf_counter()
        def progress(s):
            print(f'  samples={len(s.points):,} certified_cells={len(s.cert_faces):,} '
                  f'pending={len(s.pending):,} elapsed={time.perf_counter() - tick:.1f}s', flush=True)
        if index < len(manifest['completed']):
            report = certify_model(directory / 'model.ply', directory / 'surface', args.quality)
            report['quality'] = sampler.validate_quality(directory / 'model.ply')
            report.update(geometry_sha256=sampler.geometry_hash, material_sha256=sampler.material_hash,
                          quality_protocol=asdict(args.quality), skipped_degenerate_faces=sampler.surface.skipped,
                          count_stop_condition=False, processed_cells=sampler.processed)
            if report['quality']['status'] != 'complete':
                report['status'] = 'incomplete'
            boundary = np.any(sampler.boundary_corner[np.asarray(sampler.faces)] & (np.asarray(sampler.bary) == 0), axis=1)
            np.save(directory / 'surface' / 'boundary.npy', boundary)
            np.save(directory / 'surface' / 'repair_reason.npy', np.where(boundary, 'boundary', 'deficit'))
            (directory / 'coverage_report.json').write_text(json.dumps(report, indent=2) + '\n')
        else:
            sampler.run(checkpoint=state, progress=progress)
            report = sampler.export(directory)
        np.save(directory / 'surface' / 'loop_triangle.npy',
                np.concatenate(group['loop_triangles'])[sampler.surface.source])
        np.save(directory / 'surface' / 'rejected_source_face.npy',
                np.setdiff1d(np.arange(sampler.original_face_count), sampler.surface.source))
        elapsed = time.perf_counter() - tick
        if index < len(manifest['completed']):
            previous_cost = manifest['completed'][index]['coverage']
            report['initialization_seconds'] = previous_cost.get('initialization_seconds', previous_cost.get('seconds'))
            report['revalidation_seconds'] = elapsed
            report['seconds'] = report['initialization_seconds']
        else:
            report['seconds'] = elapsed
            report['initialization_seconds'] = elapsed
        entry = dict(directory=str(directory.relative_to(output)), object=group['object_name'],
                     material=desc['name'], colour=desc['color'], has_uv=uv is not None,
                     image=None if desc['image'] is None else desc['image'].name, coverage=report)
        if index < len(manifest['completed']):
            manifest['completed'][index] = entry
        else:
            manifest['completed'].append(entry)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        if report['status'] != 'complete':
            previous_report = output / 'coverage_report.json'
            if previous_report.exists():
                failed = json.loads(previous_report.read_text())
                failed.update(status='incomplete', failed_group=index, failed_group_report=report)
                previous_report.write_text(json.dumps(failed, indent=2) + '\n')
            raise SystemExit('Export failed continuous coverage certification; checkpoint preserved')
    report = combine(output, manifest['completed'], source_hash, args, bounds)
    manifest['status'] = report['status']
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{report["status"]}: {report["gaussians"]:,} Gaussians; '
          f'minimum certified coverage {report["minimum_lower_bound"]:.8f}', flush=True)


if __name__ == '__main__':
    main()
