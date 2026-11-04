"""Initialize a Blender scene with finite Mesh probes, without continuous certification.

All nondegenerate surfaces are retained. Groups are processed separately and
written to disk; a failed validation keeps the model and failing probes and
returns a nonzero exit status. Existing output directories are never replaced.
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
from mesh_to_gs.coverage import digest
from mesh_to_gs.sampled_coverage import SampledConfig, SampledSurface


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--objects', help='Object-name regex for an isolated experiment; default: entire scene')
    for name, default in asdict(SampledConfig()).items():
        parser.add_argument('--' + name.replace('_', '-'), type=type(default), default=default)
    args = parser.parse_args(argv)
    try:
        args.quality = SampledConfig(**{k: getattr(args, k) for k in asdict(SampledConfig())})
        if args.objects is not None:
            re.compile(args.objects)
    except (ValueError, re.error) as error:
        parser.error(str(error))
    return args


def combine(output, groups):
    """Copy the validated per-group float64 PLY bodies without conversion."""
    from plyfile import PlyData
    total = sum(g['report']['gaussians'] for g in groups)
    first = output / groups[0]['directory'] / 'model.ply'
    names = PlyData.read(first)['vertex'].data.dtype.names
    header = f'ply\nformat binary_little_endian 1.0\nelement vertex {total}\n'
    header += ''.join(f'property double {name}\n' for name in names) + 'end_header\n'
    with (output/'model.ply').open('wb') as stream:
        stream.write(header.encode())
        for group in groups:
            with (output/group['directory']/'model.ply').open('rb') as source:
                while True:
                    line = source.readline()
                    if not line:
                        raise ValueError('Truncated group PLY header')
                    if line.strip() == b'end_header':
                        break
                shutil.copyfileobj(source, stream, length=1 << 20)
    return total


def build(args):
    import bpy
    from mesh_to_gs.blend import extract_scene, load_texture
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit('Use a new output directory; existing models are never overwritten')
    source_hash = digest(args.input)
    output.mkdir(parents=True)
    manifest = dict(status='incomplete', method='sampled_surface_coverage', continuous_certificate=False,
                    input=str(args.input.resolve()), source_sha256=source_hash,
                    configuration=asdict(args.quality), groups=[],
                    scope='entire scene' if args.objects is None else f'object regex: {args.objects}',
                    hidden_regions_removed=0,
                    visibility='All nondegenerate source surfaces retained; no hidden-layer proof')
    report_path = output/'sampled_coverage_report.json'
    def save():
        report_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    save()
    tick = time.perf_counter()
    try:
        bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()), load_ui=False)
        groups, bounds, _ = extract_scene()
        selected = [g for g in groups.values() if args.objects is None or re.search(args.objects, g['object_name'])]
        if not selected:
            raise ValueError('No objects matched')
        manifest.update(source_bounds=bounds, target_groups=len(selected), skipped_degenerate_faces=0)
        cache = TextureCache(load_texture)
        for index, group in enumerate(selected):
            desc = group['description']
            directory = output/'groups'/f'{index:06d}'
            triangles = np.concatenate(group['triangles'])
            cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            area2 = np.hypot(np.hypot(cross[:, 0], cross[:, 1]), cross[:, 2])
            if not np.any(np.isfinite(area2) & (area2 > 0)):
                manifest['skipped_degenerate_faces'] += len(triangles)
                save()
                continue
            sampler = SampledSurface(triangles, np.concatenate(group['vertex_ids']),
                uv=np.concatenate(group['uvs']) if group['uvs'] else None,
                texture=cache.get(desc['image']), colour=desc['color'], config=args.quality)
            print(f'Group {index + 1}/{len(selected)}: {group["object_name"]} / {desc["name"]}', flush=True)
            group_tick = time.perf_counter()
            sampler.fit(progress=lambda event: print(
                f'  round={event["round"]} samples={event["gaussians"]:,} '
                f'failed_fit_probes={event["failed_probes"]:,}', flush=True))
            report = sampler.export(directory)
            np.save(directory/'loop_triangle.npy', np.concatenate(group['loop_triangles'])[sampler.surface.source])
            report['seconds'] = time.perf_counter() - group_tick
            manifest['skipped_degenerate_faces'] += sampler.surface.skipped
            manifest['groups'].append(dict(directory=str(directory.relative_to(output)),
                object=group['object_name'], material=desc['name'], report=report))
            save()
            print(f'  independent probes: {report["status"]}; '
                  f'{report["validation"]["failed_probes"]}/{report["validation"]["probes"]} failed', flush=True)
        if not manifest['groups']:
            raise ValueError('No nondegenerate surface to sample')
        manifest['gaussians'] = combine(output, manifest['groups'])
        reports = [g['report'] for g in manifest['groups']]
        manifest.update(status='passed' if all(r['status'] == 'passed' for r in reports) else 'failed',
                        model_sha256=digest(output/'model.ply'),
                        target_faces=sum(r['target_faces'] for r in reports),
                        target_area=sum(r['target_area'] for r in reports),
                        validation_stage='per-group serialized float64 PLY; combined body copied byte-for-byte',
                        seconds=time.perf_counter() - tick)
        save()
    except BaseException as error:
        manifest.update(status='incomplete', error=f'{type(error).__name__}: {error}', seconds=time.perf_counter() - tick)
        save()
        raise
    return manifest


def main():
    report = build(parse_args())
    if report['status'] != 'passed':
        raise SystemExit('Finite-probe validation failed; model and failed probes retained for diagnosis')


if __name__ == '__main__':
    main()
