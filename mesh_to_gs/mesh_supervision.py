"""Mesh surface/colour queries and verified empty-space labels for joint refinement."""
import hashlib
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def model_inputs(model_dir):
    """Load the single attached-initializer schema."""
    model_dir = Path(model_dir)
    report = json.loads((model_dir / 'initialization_report.json').read_text())
    # Octree initialization deliberately performs no visibility filtering.
    # Training still uses the existing scene-aware close-layer query policy.
    fallback = {'mode': 'auto', 'max_gap': .007} if report.get('method') == 'octree-voxel-surface' else {'mode': 'open', 'max_gap': 0.}
    report = dict(report, visibility=report.get('visibility', fallback))
    return report, model_dir / 'surface_metadata'


def surface_distribution(triangles, charts, row_faces, retain_all=False, mode='density'):
    """Half area sampling, half initializer-density sampling; no semantic masks.

    Area sampling includes unsampled triangles in represented charts. Charts
    entirely removed by visibility are excluded, since the fixed rows cannot
    represent them. The density half preserves supervision of small details.
    """
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    if retain_all:
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        area = np.hypot(np.hypot(cross[:, 0], cross[:, 1]), cross[:, 2]) / 2
    else:
        represented = np.unique(charts[row_faces])
        area *= np.isin(charts, represented)
    count = np.bincount(row_faces, minlength=len(triangles)).astype(np.float64)
    if area.sum() <= 0 or count.sum() <= 0:
        raise ValueError('No represented nondegenerate surface')
    if mode == 'area':
        return area / area.sum()
    if mode != 'density':
        raise ValueError('Unknown surface query mode')
    return .5 * area / area.sum() + .5 * count / count.sum()


def sample_surface(triangles, probability, count, rng):
    face = np.searchsorted(np.cumsum(probability), rng.random(count), side='right')
    face = face.clip(max=len(triangles) - 1)
    r = rng.random((count, 2))
    a = np.sqrt(r[:, 0])
    bary = np.column_stack((1 - a, a * (1 - r[:, 1]), a * r[:, 1]))
    return np.einsum('ni,nij->nj', bary, triangles[face]), face, bary


def mesh_colours(metadata, faces, barycentric, groups, blend):
    """Query the exact cached face-corner UV/material, in linear RGB."""
    import bpy
    from .adaptive import bilinear
    from .blend import load_texture
    from .color import srgb_to_linear
    from .surface import linear_texture
    bpy.ops.wm.open_mainfile(filepath=str(Path(blend).resolve()), load_ui=False)
    uv = np.load(metadata / 'uv.npy', mmap_mode='r')
    gi = np.searchsorted([g['faces'][1] for g in groups], faces, side='right')
    colour = np.empty((len(faces), 3), np.float32)
    images = OrderedDict()
    cached_bytes = 0
    for index in np.unique(gi):
        g = groups[index]
        rows = np.flatnonzero(gi == index)
        if g['image']:
            key = g['image']
            if key in images:
                image = images.pop(key)
            else:
                image = linear_texture(load_texture(bpy.data.images[key])).astype(np.float32)
                cached_bytes += image.nbytes
            images[key] = image
            while cached_bytes > 512 * 1024 ** 2 and len(images) > 1:
                _, evicted = images.popitem(last=False)
                cached_bytes -= evicted.nbytes
            coords = np.einsum('ni,nij->nj', barycentric[rows], uv[faces[rows]])
            colour[rows] = bilinear(image, coords)
        else:
            colour[rows] = srgb_to_linear(np.asarray(g['color'], np.float32))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return colour


def local_queries(points, means, quats, scales, tree, k, tolerance):
    """Subtract in float64 before converting to unit-scaled local float32.

    This avoids cancellation in world coordinates ~100 m. A fixed normal
    tolerance accounts for source/PLY precision without permitting tangent
    support to relax it. The same candidates are used throughout training.
    """
    _, ids = tree.query(points, k=min(k, len(means)), workers=8)
    ids = np.asarray(ids).reshape(len(points), -1)
    rotation = Rotation.from_quat(quats[ids].reshape(-1, 4), scalar_first=True).as_matrix()
    delta = (points[:, None] - means[ids]).reshape(-1, 3)
    local = np.einsum('ni,nij->nj', delta, rotation).reshape(len(points), -1, 3)
    units = scales[ids].copy()
    units[..., 2] = np.hypot(units[..., 2], tolerance)
    return ids.astype(np.int32), (local / units).astype(np.float32)


def potential_candidates(points, means, quats, scales, tree, k, tolerance):
    """Select fixed candidates by their maximum support under allowed changes.

    A larger Euclidean shortlist avoids losing broad splats behind densely
    sampled tiny ones. The score allows 1.5x scales and +/-45 degree rotation;
    selection is performed once, outside autograd. It remains approximate.
    """
    ids, local = local_queries(points, means, quats, scales, tree, min(8 * k, len(means)), tolerance)
    sigma = scales[ids]
    x, y = local[..., 0] * sigma[..., 0], local[..., 1] * sigma[..., 1]
    preferred = np.arctan2(y, x) - np.where(sigma[..., 0] >= sigma[..., 1], 0., np.pi / 2)
    angle = (preferred + np.pi / 2) % np.pi - np.pi / 2
    angle = angle.clip(-np.pi / 4, np.pi / 4)
    c, s = np.cos(angle), np.sin(angle)
    a = (x * c + y * s) / (1.5 * sigma[..., 0])
    b = (-x * s + y * c) / (1.5 * sigma[..., 1])
    z = local[..., 2] * np.hypot(sigma[..., 2], tolerance) / np.hypot(1.5 * sigma[..., 2], tolerance)
    energy = a * a + b * b + z * z
    n = min(k, len(means))
    selected = np.argpartition(energy, n - 1, axis=1)[:, :n]
    return np.take_along_axis(ids, selected, axis=1), np.take_along_axis(local, selected[..., None], axis=1)


def upgrade_neighbours(model_dir, cache, saved):
    """Reuse exact GT samples while upgrading an older Euclidean-only cache."""
    import time
    from numpy.lib.format import open_memmap
    from plyfile import PlyData
    start = time.perf_counter()
    config = saved['configuration']
    data = PlyData.read(model_dir / 'model.ply', mmap='r')['vertex'].data
    means = np.column_stack([data[x] for x in 'xyz']).astype(np.float64)
    quats = np.column_stack([data[f'rot_{i}'] for i in range(4)]).astype(np.float64)
    scales = np.exp(np.column_stack([data[f'scale_{i}'] for i in range(3)]).astype(np.float64))
    points = np.load(cache / 'points.npy', mmap_mode='r')
    tree = cKDTree(means)
    k, tolerance = config['k'], config['tolerance']
    out_ids = open_memmap(cache / 'neighbours_new.npy', 'w+', np.int32, (len(points), min(k, len(means))))
    out_local = open_memmap(cache / 'local_new.npy', 'w+', np.float32, (len(points), min(k, len(means)), 3))
    for a in range(0, len(points), 4096):
        b = min(a + 4096, len(points))
        out_ids[a:b], out_local[a:b] = potential_candidates(points[a:b], means, quats, scales, tree, k, tolerance)
        if a % 262144 == 0:
            print(f'potential neighbours {b}/{len(points)}; {time.perf_counter() - start:.1f}s', flush=True)
    out_ids.flush(); out_local.flush()
    rng = np.random.default_rng(config['seed'] + 1)
    probes = rng.choice(len(points), 1024, replace=False)
    wide_ids, wide_local = local_queries(points[probes], means, quats, scales, tree, min(16 * k, len(means)), tolerance)
    selected = out_ids[probes]
    missing = ~(wide_ids[..., None] == selected[:, None, :]).any(-1)
    opacity = 1 / (1 + np.exp(-data['opacity'].astype(np.float64)))
    u = opacity[wide_ids] * np.exp(-.5 * (wide_local.astype(np.float64) ** 2).sum(-1))
    omitted = (u * missing).sum(-1)
    full_alpha = -np.expm1(np.log1p(-u).sum(-1))
    kept_alpha = -np.expm1(np.log1p(-u * ~missing).sum(-1))
    saved['previous_neighbour_audit'] = saved['neighbour_audit']
    saved['neighbour_audit'] = dict(probes=len(probes), comparison_k=wide_ids.shape[1],
                                    mean_omitted_weight=float(omitted.mean()), max_omitted_weight=float(omitted.max()),
                                    fraction_omitted_over_1e_3=float((omitted > 1e-3).mean()),
                                    mean_alpha_error=float((full_alpha - kept_alpha).mean()),
                                    max_alpha_error=float((full_alpha - kept_alpha).max()))
    del out_ids, out_local
    (cache / 'neighbours_new.npy').replace(cache / 'neighbours.npy')
    (cache / 'local_new.npy').replace(cache / 'local.npy')
    saved['configuration']['version'] = 2
    saved['neighbour_policy'] = 'Top K potential contributors (1.5x scale, +/-45deg) from nearest 8*K; fixed throughout training'
    saved['neighbour_upgrade_seconds'] = time.perf_counter() - start
    for filename in ('local.npy', 'neighbours.npy'):
        saved['files'][filename] = digest(cache / filename)
    (cache / 'queries.json').write_text(json.dumps(saved, indent=2) + '\n')
    return saved




def prepare_queries(model_dir, cache, count=1 << 20, k=64, tolerance=1e-4, seed=20261102,
                    retain_all=False, surface_query_mode='density', filter_hidden=False):
    """Build reusable visible surface / verified off-surface query data."""
    import time
    from numpy.lib.format import open_memmap
    from plyfile import PlyData
    from .visibility import VisibilityTester, mesh_distance
    model_dir, cache = Path(model_dir), Path(cache)
    report, metadata = model_inputs(model_dir)
    configuration = dict(version=1, model_sha256=digest(model_dir / 'model.ply'),
                         source_sha256=digest(report['source']), count=count, k=k,
                         tolerance=tolerance, seed=seed)
    if retain_all:
        configuration['retain_all_surfaces'] = True
    if surface_query_mode != 'density':
        configuration['surface_query_mode'] = surface_query_mode
    if filter_hidden:
        configuration['filter_hidden_surface'] = True
    if configuration['source_sha256'] != report['input_sha256']:
        raise ValueError('Source Mesh changed since initialization')
    manifest = cache / 'queries.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        compared = dict(saved['configuration'], version=1)
        if compared != configuration:
            raise ValueError('Query cache does not match model or sampling configuration')
        if saved['configuration']['version'] == 1:
            return upgrade_neighbours(model_dir, cache, saved)
        return saved
    if cache.exists() and any(cache.iterdir()):
        raise ValueError('Incomplete query cache: use a new directory')
    cache.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    data = PlyData.read(model_dir / 'model.ply', mmap='r')['vertex'].data
    means = np.column_stack([data[x] for x in 'xyz']).astype(np.float64)
    quats = np.column_stack([data[f'rot_{i}'] for i in range(4)]).astype(np.float64)
    scales = np.exp(np.column_stack([data[f'scale_{i}'] for i in range(3)]).astype(np.float64))
    triangles = np.load(metadata / 'triangles.npy', mmap_mode='r')
    charts = np.load(metadata / 'chart.npy', mmap_mode='r')
    row_faces = np.load(metadata / 'face.npy', mmap_mode='r')
    probability = surface_distribution(triangles, charts, row_faces, retain_all, surface_query_mode)
    tester = VisibilityTester.for_scene(triangles, report['source_bounds'],
                                       report.get('visibility', {}).get('mode', 'auto'))
    rng = np.random.default_rng(seed)
    points, faces, bary = [], [], []
    kept = rejected = 0
    while kept < count:
        n = min(max(count - kept, 65536), 262144)
        p, f, b = sample_surface(triangles, probability, n, rng)
        cross = np.cross(triangles[f, 1] - triangles[f, 0], triangles[f, 2] - triangles[f, 0])
        normals = cross / np.linalg.norm(cross, axis=1)[:, None]
        take = np.ones(n, bool)
        if filter_hidden or not retain_all:
            near_layer = tester.normal_gap(p, normals, report['visibility']['max_gap']) < report['visibility']['max_gap']
            if near_layer.any():
                take[near_layer] = tester.visible(p[near_layer])
        rejected += int((~take).sum())
        ids = np.flatnonzero(take)[:count - kept]
        points.append(p[ids]); faces.append(f[ids]); bary.append(b[ids])
        kept += len(ids)
        print(f'surface queries {kept}/{count}; {time.perf_counter() - start:.1f}s', flush=True)
        if rejected > 20 * count:
            raise ValueError('Insufficient visible surface for query sampling')
    points, faces, bary = map(np.concatenate, (points, faces, bary))
    groups = json.loads((metadata / 'groups.json').read_text())
    colour = mesh_colours(metadata, faces, bary, groups, report['source'])
    np.save(cache / 'colour.npy', colour)
    np.save(cache / 'surface_face.npy', faces.astype(np.int32))
    np.save(cache / 'surface_barycentric.npy', bary)
    # Tangent candidates expose open edges and holes; normal-band candidates
    # expose thickness. Reject points close to ANY triangle, not just the seed.
    empty_points = []
    empty_count = empty_candidates = 0
    boundary_rows = np.flatnonzero(np.load(metadata / 'boundary.npy', mmap_mode='r'))
    while empty_count < count:
        n = min(max(2 * (count - empty_count), 65536), 262144)
        rows = rng.integers(len(means), size=n)
        if len(boundary_rows):
            rows[:n // 3] = rng.choice(boundary_rows, n // 3)
        rotation = Rotation.from_quat(quats[rows], scalar_first=True).as_matrix()
        offset = rng.normal(size=(n, 3))
        offset[:, :2] *= scales[rows, :2] * 1.5
        band = np.hypot(scales[rows, 2], tolerance)
        offset[:, 2] = rng.choice([-1, 1], n) * rng.uniform(1.5, 3., n) * band
        offset[:n // 2, 2] = 0
        candidates = means[rows] + np.einsum('nij,nj->ni', rotation, offset)
        distance = mesh_distance(tester, candidates, max_distance=1.)
        selected = np.flatnonzero(distance > 2 * tolerance)[:count - empty_count]
        empty_points.append(candidates[selected])
        empty_count += len(selected)
        empty_candidates += n
        print(f'empty queries {empty_count}/{count}; {time.perf_counter() - start:.1f}s', flush=True)
        if empty_candidates > 100 * count:
            raise ValueError('Insufficient off-surface queries')
    empty_points = np.concatenate(empty_points)
    del tester, probability
    tree = cKDTree(means)
    candidates = min(k, len(means))
    ids_out = open_memmap(cache / 'neighbours.npy', mode='w+', dtype=np.int32, shape=(2 * count, candidates))
    local_out = open_memmap(cache / 'local.npy', mode='w+', dtype=np.float32, shape=(2 * count, candidates, 3))
    all_points = np.concatenate((points, empty_points))
    np.save(cache / 'points.npy', all_points)
    for a in range(0, len(all_points), 16384):
        b = min(a + 16384, len(all_points))
        ids_out[a:b], local_out[a:b] = local_queries(all_points[a:b], means, quats, scales, tree, k, tolerance)
        if a % 262144 == 0:
            print(f'neighbours {b}/{len(all_points)}; {time.perf_counter() - start:.1f}s', flush=True)
    ids_out.flush(); local_out.flush()
    # Audit omitted contributions at a larger K on independent query rows.
    probe = rng.choice(count, min(2048, count), replace=False)
    wide_k = min(max(4 * k, 128), len(means))
    wide_ids, wide_local = local_queries(points[probe], means, quats, scales, tree, wide_k, tolerance)
    opacity = 1 / (1 + np.exp(-data['opacity'].astype(np.float64)))
    u = opacity[wide_ids] * np.exp(-.5 * np.sum(wide_local.astype(np.float64) ** 2, axis=-1))
    omitted = u[:, candidates:].sum(-1)
    saved = dict(configuration=configuration, surface_queries=count, empty_queries=count,
                 gaussians=len(data), rejected_hidden_surface_queries=rejected,
                 empty_candidates=empty_candidates, seconds=time.perf_counter() - start,
                 sampling=('100% retained surface area' if surface_query_mode == 'area' else
                           '50% chart area / 50% initializer face density') + ('; all source faces retained' if retain_all else '; hidden close layers filtered'),
                 empty_rule='Tangent and normal-band probes; unsigned distance to full Mesh > 2*tolerance',
                 neighbour_audit=dict(probes=len(probe), comparison_k=wide_k,
                                      mean_omitted_weight=float(omitted.mean()),
                                      max_omitted_weight=float(omitted.max()),
                                      fraction_omitted_over_1e_3=float((omitted > 1e-3).mean())),
                 files={p.name: digest(p) for p in cache.glob('*.npy')})
    manifest.write_text(json.dumps(saved, indent=2) + '\n')
    return upgrade_neighbours(model_dir, cache, saved)


def verify_empty_queries(model_dir, cache):
    """Validate every negative label against all original float64 GT triangles."""
    import time
    from .visibility import VisibilityTester, mesh_near_surface
    model_dir, cache = Path(model_dir), Path(cache)
    manifest = cache / 'queries.json'
    saved = json.loads(manifest.read_text())
    if 'empty_label_verification' in saved:
        return saved
    start = time.perf_counter()
    report, metadata = model_inputs(model_dir)
    triangles = np.load(metadata / 'triangles.npy', mmap_mode='r')
    tester = VisibilityTester.for_scene(triangles, report['source_bounds'], report['visibility']['mode'])
    points = np.load(cache / 'points.npy', mmap_mode='r')
    n = saved['surface_queries']
    invalid = mesh_near_surface(tester, triangles, points[n:], 2 * saved['configuration']['tolerance'])
    valid = np.flatnonzero(~invalid).astype(np.int32)
    if len(valid) < 8192:
        raise ValueError('Too few verified empty queries')
    np.save(cache / 'valid_empty.npy', valid)
    saved['empty_label_verification'] = dict(method='Conservative full-Mesh AABB query + original float64 triangle/segment distance',
                                             checked=len(invalid), removed=int(invalid.sum()), valid=len(valid),
                                             seconds=time.perf_counter() - start,
                                             tolerance_world_units=2 * saved['configuration']['tolerance'])
    saved['files']['valid_empty.npy'] = digest(cache / 'valid_empty.npy')
    manifest.write_text(json.dumps(saved, indent=2) + '\n')
    print(json.dumps(saved['empty_label_verification']), flush=True)
    return saved
