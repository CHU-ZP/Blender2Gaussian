"""Apply the selected strength-0.5 texture covariance to a surface initializer.

Preserves every position, colour, opacity, normal thickness and metadata row.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from mesh_to_gs.adaptive import TextureCache, uv_jacobians
from mesh_to_gs.surface import coplanar_seam_edges
from mesh_to_gs.support_constraints import chart_boundaries, constraints, project
from mesh_to_gs.texture_guidance import structure_tensor, guided_covariance, covariance_parameters


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def columns(data, names):
    return np.column_stack([data[n] for n in names])


def group_geometry(metadata, group):
    a, b = group['faces']
    triangles = np.asarray(metadata.triangles[a:b])
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    length = np.linalg.norm(cross, axis=1)
    return SimpleNamespace(triangles=triangles, uv=metadata.uv[a:b],
                           areas=length/2, normals=cross/length[:, None]), metadata.chart[a:b]


def group_limits(surface, chart, face, points, rotation, sigma, report, active, relax=True,
                 seam_mask=None, max_aspect=2.):
    """Index all edges of affected charts, including tiny-chart fast-path rows.

    Unless a full-group seam_mask is supplied, seams to inactive charts or
    other groups are conservatively assigned the ordinary multiple, then
    relaxed to the baseline support. Cached initializer edge
    flags are unavailable; this can suppress a gain but never tightens an
    unchanged baseline row. Querying only affected charts avoids unrelated
    tiny charts displacing relevant segments from the nearest-neighbour list.
    """
    charts = np.unique(chart[face[active]])
    keep = np.isin(chart, charts)
    s = SimpleNamespace(triangles=surface.triangles[keep], normals=surface.normals[keep])
    labels = chart[keep]
    seam = (coplanar_seam_edges(s.triangles, labels, s.normals, report['crease_degrees'])
            if seam_mask is None else seam_mask[keep])
    boundaries = chart_boundaries(s.triangles, labels, seam)
    radius = 2 * np.maximum(sigma[active, 0], sigma[active, 1]) * np.sqrt(max_aspect) + 1e-4
    m, limit = constraints(points[active], rotation[active, :, 2], rotation[active, :, 0],
                           rotation[active, :, 1], chart[face[active]], boundaries, radius,
                           multiple=report['boundary_sigmas'], seam_multiple=report['seam_sigmas'],
                           chunk=32768, chart_local=True)
    initial = np.sqrt(np.einsum('nki,ni->nk', m.astype(float)**2, sigma[active, :2]**2))
    if relax:
        limit = np.maximum(limit, initial * (1+1e-6))
    return m, limit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--strength', type=float, default=.5)
    parser.add_argument('--max-aspect', type=float, default=2.)
    args = parser.parse_args()
    if not 0 < args.strength <= 1 or args.max_aspect < 1:
        parser.error('strength must be in (0,1]; max-aspect >= 1')
    if args.output_dir.exists():
        parser.error('Use a new output directory; existing models are never overwritten')
    args.output_dir.mkdir(parents=True)
    start = time.perf_counter()
    report = json.loads((args.input/'initialization_report.json').read_text())
    path = args.input/'surface_metadata'
    metadata = SimpleNamespace(**{k: np.load(path/f'{k}.npy', mmap_mode='r')
                                 for k in ('triangles', 'uv', 'chart', 'face', 'barycentric', 'boundary')})
    groups = json.loads((path/'groups.json').read_text())
    data = PlyData.read(args.input/'model.ply', mmap='r')['vertex'].data
    outputs = [np.array(data)]
    stats = [dict(changed_rows=0, gated_rows=0, constrained_rows=0, area_ratio_sum=0.,
                  min_area_ratio=1., max_boundary_excess=0.) for _ in range(1)]
    # Opening the source only decodes named textures; evaluated Mesh is cached.
    import bpy
    from mesh_to_gs.blend import load_texture
    if digest(Path(report['source'])) != report['input_sha256']:
        raise ValueError('Source Blend changed since baseline initialization')
    bpy.ops.wm.open_mainfile(filepath=str(Path(report['source']).resolve()), load_ui=False)
    cache = TextureCache(load_texture)
    ends = np.array([g['faces'][1] for g in groups])
    row_group = np.searchsorted(ends, metadata.face, side='right')
    order = np.argsort(row_group, kind='stable')
    split = np.r_[0, np.cumsum(np.bincount(row_group, minlength=len(groups)))]
    for gi, group in enumerate(groups):
        if not group['image'] or split[gi] == split[gi+1]:
            continue
        rows = order[split[gi]:split[gi+1]]
        texture = cache.get(bpy.data.images[group['image']])
        # Most nominal textures in sk_00918 are actually constant swatches.
        if np.max(texture.image)-np.min(texture.image) < 1e-7:
            continue
        surface, chart = group_geometry(metadata, group)
        face = metadata.face[rows] - group['faces'][0]
        bary = metadata.barycentric[rows]
        bary = np.column_stack((bary, 1-bary.sum(1)))
        uv = np.einsum('ni,nij->nj', bary, surface.uv[face])
        rotation = Rotation.from_quat(columns(data[rows], [f'rot_{i}' for i in range(4)]),
                                      scalar_first=True).as_matrix()
        sigma = np.exp(columns(data[rows], [f'scale_{i}' for i in range(3)]).astype(float))
        chart_area = np.bincount(chart-chart.min(), weights=surface.areas)
        chart_count = np.bincount(chart[face]-chart.min(), minlength=len(chart_area))
        spacing = np.sqrt(chart_area[chart[face]-chart.min()] / chart_count[chart[face]-chart.min()])
        jac = uv_jacobians(surface)
        tensor = np.empty((len(rows), 2, 2), float)
        for a in range(0, len(rows), 32768):
            b = min(a+32768, len(rows))
            tensor[a:b] = structure_tensor(texture, uv[a:b], jac[face[a:b]], rotation[a:b, :, 0],
                                           rotation[a:b, :, 1], spacing[a:b])
        candidates = [guided_covariance(sigma[:, :2], tensor, spacing, s, args.max_aspect)
                      for s in (args.strength,)]
        # Use the full-strength gate for chart queries, as in the retained initialization.
        active = np.flatnonzero(candidates[0][1]/args.strength > 1e-5)
        if not len(active):
            continue
        points = columns(data[rows], 'xyz').astype(float)
        m, limit = group_limits(surface, chart, face, points, rotation, sigma, report, active,
                                max_aspect=args.max_aspect)
        for out, stat, (C, gate) in zip(outputs, stats, candidates):
            projected = project(C[active], m, limit)
            scales, quats = covariance_parameters(projected, rotation[active], sigma[active, 2])
            changed = gate[active] > 1e-5
            ids = rows[active[changed]]
            for i in range(2):
                out[f'scale_{i}'][ids] = np.log(scales[changed, i])
            for i in range(4):
                out[f'rot_{i}'][ids] = quats[changed, i]
            area_ratio = scales[:, 0]*scales[:, 1]/(sigma[active, 0]*sigma[active, 1])
            variance = np.einsum('nki,nij,nkj->nk', m, projected, m)
            stat['max_boundary_excess'] = max(stat['max_boundary_excess'],
                float(np.maximum(np.sqrt(np.maximum(variance, 0))/limit-1, 0).max(initial=0)))
            stat['changed_rows'] += len(ids)
            stat['gated_rows'] += len(active)
            stat['constrained_rows'] += int(np.isfinite(limit).any(1).sum())
            stat['area_ratio_sum'] += float(area_ratio.sum())
            stat['min_area_ratio'] = min(stat['min_area_ratio'], float(area_ratio.min()))
        if gi % 20 == 0 or len(rows) > 10000:
            print(f'group {gi+1}/{len(groups)} {group["object"]}: {len(active)}/{len(rows)} guided; '
                  f'{time.perf_counter()-start:.1f}s', flush=True)
    for strength, out, stat in zip((args.strength,), outputs, stats):
        output = args.output_dir
        held = [k for k in data.dtype.names if not (k.startswith('rot_') or k in ('scale_0', 'scale_1'))]
        for k in held:
            if not np.array_equal(data[k], out[k]):
                raise ValueError(f'Control violated: {k}')
        if not all(np.isfinite(out[k]).all() for k in out.dtype.names):
            raise ValueError('Nonfinite PLY')
        PlyData([PlyElement.describe(out, 'vertex')]).write(output/'model.ply')
        (output/'surface_metadata').symlink_to(path.resolve(), target_is_directory=True)
        shutil.copyfile(args.input/'initialization_report.json', output/'initialization_report.json')
        stat['mean_area_ratio_changed'] = stat.pop('area_ratio_sum')/max(stat['gated_rows'], 1)
        result = dict(method='texture_covariance', input=str(args.input.resolve()), input_sha256=digest(args.input/'model.ply'),
                      model_sha256=digest(output/'model.ply'), gaussians=len(out), strength=strength,
                      max_aspect=args.max_aspect, held_bit_identical=held, visibility_mask='baseline rows frozen',
                      seconds=time.perf_counter()-start, stats=stat,
                      boundary_policy='Chart-local nearest-8 limits, relaxed to baseline support; see group_limits docstring')
        (output/'covariance_report.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    source = args.output_dir/'source'
    source.mkdir()
    for name in ('scripts/apply_texture_covariance.py', 'mesh_to_gs/texture_guidance.py',
                 'mesh_to_gs/adaptive.py', 'mesh_to_gs/support_constraints.py'):
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).resolve().parents[1]/name, target)


if __name__ == '__main__':
    main()
