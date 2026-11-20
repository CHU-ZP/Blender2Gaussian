"""Lightweight geometry-only selection of complementary training cameras."""
import numpy as np
from scipy.sparse import csr_matrix

from .cameras import camera_entry, look_at, pixel_rays
from .visibility import mesh_distance


def surface_targets(triangles, groups, count, rng):
    """Mix area and sqrt(group area) sampling, preserving small object groups."""
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=1)/2
    ends = np.asarray([g['faces'][1] for g in groups])
    starts = np.r_[0, ends[:-1]]
    if ends[-1] != len(triangles) or any(g['faces'][0] != s for g, s in zip(groups, starts)):
        raise ValueError('Expected contiguous complete Mesh groups')
    group_area = np.add.reduceat(area, starts)
    if np.any(group_area <= 0):
        raise ValueError('Require positive-area groups')
    group_probability = .5*group_area/group_area.sum()+.5*np.sqrt(group_area)/np.sqrt(group_area).sum()
    probability = area*np.repeat(group_probability/group_area, ends-starts)
    faces = np.searchsorted(np.cumsum(probability), rng.random(count), side='right').clip(max=len(area)-1)
    uv = rng.random((count, 2)); r = np.sqrt(uv[:, 0])
    bary = np.c_[1-r, r*(1-uv[:, 1]), r*uv[:, 1]]
    return np.einsum('ni,nij->nj', bary, triangles[faces]), np.searchsorted(ends, faces, side='right')


def distinct_pose(camera, others, position_radius=.4, angle_degrees=20.):
    world = np.asarray(camera['camera_to_world'])
    if not others:
        return True
    matrices = np.asarray([c['camera_to_world'] for c in others])
    close = np.linalg.norm(matrices[:, :3, 3]-world[:3, 3], axis=1) < position_radius
    aligned = matrices[:, :3, 2]@world[:3, 2] > np.cos(np.deg2rad(angle_degrees))
    return not np.any(close & aligned)


def candidate_views(tester, triangles, targets, target_groups, base, count, rng,
                    width=1280, height=720):
    eyes = tester.viewpoints[tester.viewpoint_clearance >= .35]
    eyes = eyes[mesh_distance(tester, eyes, 1.) >= .35]
    if tester.mode == 'indoor':
        low = triangles.reshape(-1, 3).min(0)
        eye_height = eyes[:, 2]-low[2]
        eyes = eyes[(eye_height >= .65) & (eye_height <= 2.25)]
        floor = tester.first_hit(eyes, np.broadcast_to([0., 0, -1.], eyes.shape))
        eyes = eyes[np.abs(eyes[:, 2]-floor-low[2]) <= .20]
    if not len(eyes):
        raise ValueError('No suitable free-space camera eyes')
    result = []; attempted = 0
    for _ in range(50):
        size = max(8192, count*8)
        eye = eyes[rng.integers(len(eyes), size=size)]
        ids = rng.integers(len(targets), size=size); target = targets[ids]
        offset = target-eye; distance = np.linalg.norm(offset, axis=1)
        direction = offset/np.maximum(distance, 1e-12)[:, None]
        detail = rng.random(size) < .35
        eligible = (distance >= np.where(detail, .65, 1.5)) & (distance <= np.where(detail, 3., 8.))
        eligible &= np.abs(direction[:, 2]) < .75
        if tester.mode == 'indoor':
            # New views complement the base set at human/object height; the
            # area-based score otherwise fills the pool with ceiling views.
            eligible &= target[:, 2] < low[2]+2.6
            eligible &= np.where(detail, target[:, 2] > low[2]+.08,
                                  (direction[:, 2] > -.45) & (direction[:, 2] < .35))
        selected = np.flatnonzero(eligible)
        clear = tester.first_hit(eye[selected], direction[selected]) >= distance[selected]-.015
        attempted += size
        for i in selected[clear]:
            entry = camera_entry(f'candidate_{len(result):04d}', look_at(eye[i], target[i]),
                                 width, height, 55. if detail[i] else 70., role='training',
                                 target_group=int(target_groups[ids[i]]), target=target[i].tolist(),
                                 selection_kind='detail' if detail[i] else 'context')
            if distinct_pose(entry, base) and distinct_pose(entry, result):
                result.append(entry)
            if len(result) == count:
                return result, dict(eligible_eyes=len(eyes), attempted_pairs=attempted)
    raise ValueError(f'Only {len(result)} distinct candidate cameras found')


def view_features(tester, triangles, groups, cameras, columns=48, rows=27, cell_size=.20):
    """Coarse first-hit surface cells and 8 viewing sectors, no RGB rendering."""
    rays_per_view = columns*rows; keys = []; owners = []; stats = []
    ends = np.asarray([g['faces'][1] for g in groups])
    for start in range(0, len(cameras), 64):
        part = cameras[start:start+64]
        directions = np.concatenate([pixel_rays(c, columns, rows) for c in part])
        origins = np.repeat(np.asarray([c['camera_to_world'] for c in part])[:, :3, 3], rays_per_view, axis=0)
        distance, face = tester.first_hit_face(origins, directions)
        for j, camera in enumerate(part):
            d = distance[j*rays_per_view:(j+1)*rays_per_view]
            stats.append(dict(camera=camera['name'], foreground=float(np.isfinite(d).mean()),
                              distance_quantiles=np.quantile(d[np.isfinite(d)], [.1, .5, .9]).tolist() if np.isfinite(d).any() else []))
        valid = (face >= 0) & (distance > .05) & (distance < 250.)
        p = origins[valid]+directions[valid]*distance[valid, None]
        tri = triangles[face[valid]]; n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        axis = np.abs(n).argmax(1)
        azimuth = (np.floor((np.arctan2(-directions[valid, 1], -directions[valid, 0])+np.pi)/(np.pi/2)).astype(int) % 4)
        elevation = (directions[valid, 2] < 0).astype(int)
        key = np.c_[np.searchsorted(ends, face[valid], side='right'),
                    np.floor((p-tester.origin)/cell_size).astype(int), axis, azimuth, elevation]
        keys.append(key.astype(np.int32));owners.append(np.repeat(np.arange(start, start+len(part)), rays_per_view)[valid])
    all_keys = np.concatenate(keys);owner = np.concatenate(owners)
    unique, inverse = np.unique(all_keys, axis=0, return_inverse=True)
    matrix = csr_matrix((np.ones(len(owner), np.float32), (owner, inverse)), shape=(len(cameras), len(unique)))
    matrix.data[:] = 1.  # A view counts once per feature, irrespective of pixel duplicates.
    group_features = np.bincount(unique[:, 0], minlength=len(groups))
    weights = 1./np.sqrt(group_features[unique[:, 0]])
    return matrix, weights, unique, stats


def greedy_views(matrix, weights, base_count, add_count, eligible=None, strata=None, quotas=None):
    """Diminishing returns for seen surface/direction features; stable tie order."""
    if not 0 <= base_count <= matrix.shape[0] or add_count < 0:
        raise ValueError('Invalid view counts')
    available = np.ones(matrix.shape[0], bool) if eligible is None else np.asarray(eligible, bool).copy()
    available[:base_count] = False
    if available.sum() < add_count:
        raise ValueError('Insufficient eligible camera candidates')
    quota = dict(quotas or {})
    if quota:
        strata = np.asarray(strata)
        if len(strata) != matrix.shape[0] or sum(quota.values()) != add_count:
            raise ValueError('Stratum quotas must cover all added views')
        for label, count in quota.items():
            if count < 0 or np.count_nonzero(available & (strata == label)) < count:
                raise ValueError('Insufficient candidates for stratum quota')
        available &= np.isin(strata, [label for label, count in quota.items() if count > 0])
    coverage = np.asarray(matrix[:base_count].sum(axis=0)).ravel()
    initial = coverage.copy();chosen = [];gains = []
    for _ in range(add_count):
        gains_all = np.asarray(matrix @ (weights/(1.+coverage))).ravel()
        gains_all[~available] = -np.inf
        index = int(np.argmax(gains_all))
        chosen.append(index);gains.append(float(gains_all[index]));available[index] = False
        coverage[matrix[index].indices] += 1
        if quota:
            quota[strata[index]] -= 1
            if quota[strata[index]] == 0:
                available[strata == strata[index]] = False
    return chosen, dict(gains=gains, observed_features_before=int((initial > 0).sum()),
                        observed_features_after=int((coverage > 0).sum()), feature_universe=matrix.shape[1],
                        weighted_observed_before=float(weights[initial > 0].sum()/weights.sum()),
                        weighted_observed_after=float(weights[coverage > 0].sum()/weights.sum()))
