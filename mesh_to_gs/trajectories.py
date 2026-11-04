"""Continuous, geometry-checked camera paths for retained-model visualization."""
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .cameras import camera_entry, look_at, pixel_rays


def keyframe_indices(count, closed=False):
    """Six representative frames without repeating the end of a closed orbit."""
    return (np.arange(6)*count//6 if closed else np.linspace(0, count-1, 6).round().astype(int))


def resample_path(points, count, ease=False):
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or count < 2 or not np.isfinite(points).all():
        raise ValueError('A finite Nx3 path and at least two frames are required')
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10]]
    if len(points) < 2:
        raise ValueError('The camera path must have positive length')
    distance = np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
    time = np.linspace(0, 1, count)
    if ease:
        time = time*time*(3-2*time)
    return np.column_stack([np.interp(time*distance[-1], distance, points[:, axis]) for axis in range(3)])


def dense_path(points, spacing=.05, closed=False):
    """Include intermediate positions; maximum distance between samples is spacing."""
    points = np.asarray(points, np.float64)
    if spacing <= 0 or len(points) < 2:
        raise ValueError('Positive spacing and at least two positions are required')
    end = np.roll(points, -1, axis=0) if closed else points[1:]
    start = points if closed else points[:-1]
    samples = []
    for a, b in zip(start, end):
        subdivisions = max(1, math.ceil(np.linalg.norm(b-a)/spacing))
        samples.append(a+(b-a)*np.arange(subdivisions)[:, None]/subdivisions)
    return np.concatenate([*samples, points[:1] if closed else points[-1:]])


def path_clear(tester, triangles, points, clearance=.25, spacing=.05, closed=False):
    from .visibility import mesh_near_surface
    samples = dense_path(points, spacing, closed)
    # Distance to a closed triangle is 1-Lipschitz. The half-spacing guard
    # protects the entire segment between checked samples, not just frames.
    near = mesh_near_surface(tester, triangles, samples, clearance+spacing/2+1e-5)
    return not bool(near.any()), len(samples)


def indoor_path(tester, triangles, bounds, count, seed, max_length, clearance=.25, grid=.2):
    """A smoothed geodesic through the largest floor-level free-space component."""
    from .visibility import fibonacci_directions, mesh_near_surface
    low, high = np.asarray(bounds, np.float64)
    level = low[2]+1.5
    x = np.arange(low[0]+grid/2, high[0], grid)
    y = np.arange(low[1]+grid/2, high[1], grid)
    xy = np.stack(np.meshgrid(x, y, indexing='ij'), -1).reshape(-1, 2)
    eyes = np.column_stack((xy, np.full(len(xy), level)))
    probe = tester._probe(eyes, fibonacci_directions(48), float(np.linalg.norm(high-low)))
    down = np.tile([0., 0., -1.], (len(eyes), 1))
    ground = level-tester.first_hit(eyes, down)
    valid = (probe[:, 0] >= .75) & (np.abs(ground-low[2]) <= .2)
    candidate = np.flatnonzero(valid)
    # Four-neighbour edges are grid metres long. This extra guard also
    # leaves room for the subsequent dense-path audit's half-sample guard.
    near = mesh_near_surface(tester, triangles, eyes[candidate], clearance+grid/2+.0251)
    valid[candidate[near]] = False
    shape = (len(x), len(y))
    ids = np.full(shape, -1, np.int64)
    ids.flat[np.flatnonzero(valid)] = np.arange(valid.sum())
    if valid.sum() < 10:
        raise ValueError('Too little connected free space at camera height')
    pairs = []
    for a, b in ((ids[:-1], ids[1:]), (ids[:, :-1], ids[:, 1:])):
        keep = (a >= 0) & (b >= 0)
        pairs.append(np.column_stack((a[keep], b[keep])))
    edges = np.concatenate(pairs)
    edges = np.concatenate((edges, edges[:, ::-1]))
    graph = coo_matrix((np.full(len(edges), grid), edges.T), shape=(valid.sum(), valid.sum())).tocsr()
    _, component = connected_components(graph)
    largest = np.argmax(np.bincount(component))
    available = np.flatnonzero(component == largest)
    rng = np.random.default_rng(seed)
    first = int(rng.choice(available))
    distances = dijkstra(graph, indices=first)
    first = int(available[np.argmax(distances[available])])
    distances, predecessor = dijkstra(graph, indices=first, return_predecessors=True)
    last = int(available[np.argmax(distances[available])])
    route = [last]
    while route[-1] != first:
        route.append(int(predecessor[route[-1]]))
    points = eyes[valid][route[::-1]]
    length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    if length > max_length:
        # Keep a middle section, away from the most confined graph endpoints.
        samples = resample_path(points, max(2, math.ceil(length/.02)))
        start = (length-max_length)/2
        distances = np.r_[0., np.linalg.norm(np.diff(samples, axis=0), axis=1).cumsum()]
        points = samples[(distances >= start) & (distances <= start+max_length)]
    raw = resample_path(points, count)
    smooth = gaussian_filter1d(raw, sigma=max(2., count/48), axis=0, mode='nearest')
    for blend in (1., .5, .25, .125, 0.):
        candidate = resample_path(raw*(1-blend)+smooth*blend, count, ease=True)
        clear, checks = path_clear(tester, triangles, candidate, clearance)
        if clear:
            return candidate, dict(grid_spacing=grid, camera_height=level,
                                   free_grid_points=int(valid.sum()), component_points=len(available),
                                   smoothing_blend=blend, dense_checked_positions=checks)
    raise ValueError('No interpolated indoor route passed the complete-Mesh clearance audit')


def orbit_path(bounds, count, seed, width, height, fov=70., turns=1, height_variation=0.):
    """A closed orbit fitted to keep the entire source bounding box in view."""
    if turns < 1 or int(turns) != turns or not np.isfinite(height_variation) or height_variation < 0:
        raise ValueError('Positive whole orbit turns and nonnegative finite height variation are required')
    low, high = np.asarray(bounds, np.float64)
    centre = (low+high)/2
    target = centre.copy()
    eye_z = high[2]+.75
    phase = np.random.default_rng(seed).uniform(0, 2*math.pi)
    fraction = np.arange(count)/count
    angles = phase+2*math.pi*turns*fraction
    heights = eye_z+height_variation*(1-np.cos(2*math.pi*fraction))/2
    radius = np.linalg.norm((high-low)[:2])/2+.5
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1])
                        for z in (low[2], high[2])])
    fx = width/(2*math.tan(math.radians(fov)/2))
    for _ in range(60):
        points = np.column_stack((centre[0]+radius*np.cos(angles), centre[1]+radius*np.sin(angles),
                                  heights))
        fitted = True
        for eye in points:
            matrix = look_at(eye, target)
            local = (corners-eye)@matrix[:3, :3]
            projected = np.abs(local[:, :2]/local[:, 2:]) * [2*fx/width, 2*fx/height]
            if (local[:, 2] <= .05).any() or projected.max() > .92:
                fitted = False
                break
        if fitted:
            return points, target, dict(orbit_radius=radius, camera_height=eye_z,
                                       target=target.tolist(), bounding_box_frame_margin=.08,
                                       turns=int(turns), height_variation=height_variation,
                                       minimum_camera_height=float(heights.min()),
                                       maximum_camera_height=float(heights.max()))
        radius *= 1.08
    raise ValueError('Could not fit the scene into the orbit cameras')


def make_cameras(points, width, height, fps, fov=70., target=None):
    points = np.asarray(points, np.float64)
    if target is None:
        forward = np.gradient(points, axis=0)
        yaw = gaussian_filter1d(np.unwrap(np.arctan2(forward[:, 1], forward[:, 0])),
                                sigma=max(2., len(points)/32), mode='nearest')
        pitch = math.radians(-8)
        directions = np.column_stack((np.cos(yaw)*math.cos(pitch), np.sin(yaw)*math.cos(pitch),
                                      np.full(len(points), math.sin(pitch))))
        targets = points+directions
    else:
        targets = np.broadcast_to(target, points.shape)
    return [camera_entry(f'frame_{i:06d}', look_at(eye, aim), width, height, fov,
                         frame=i, time_seconds=i/fps, role='trajectory')
            for i, (eye, aim) in enumerate(zip(points, targets))]


def audit_trajectory(tester, triangles, cameras, clearance=.25, closed=False):
    matrices = np.asarray([c['camera_to_world'] for c in cameras])
    points = matrices[:, :3, 3]
    clear, checks = path_clear(tester, triangles, points, clearance, closed=closed)
    if not clear:
        raise ValueError('Camera motion violates the minimum complete-Mesh clearance')
    rotations = matrices[:, :3, :3]
    error = float(np.abs(rotations.transpose(0, 2, 1)@rotations-np.eye(3)).max())
    if error > 1e-8 or not np.isfinite(matrices).all():
        raise ValueError('Invalid camera rotations')
    foreground = []
    for camera in cameras:
        rays = pixel_rays(camera, 24, 14)
        eye = np.asarray(camera['camera_to_world'])[:3, 3]
        foreground.append(float(np.isfinite(tester.first_hit(np.repeat(eye[None], len(rays), 0), rays)).mean()))
    if min(foreground) < .03:
        raise ValueError('A trajectory frame looks away from the scene')
    adjacent = (np.einsum('nij,nij->n', rotations, np.roll(rotations, -1, axis=0)) if closed else
                np.einsum('nij,nij->n', rotations[:-1], rotations[1:]))
    angle = np.rad2deg(np.arccos(np.clip((adjacent-1)/2, -1, 1)))
    edges = np.diff(np.r_[points, points[:1]] if closed else points, axis=0)
    return dict(status='pass', complete_mesh_clearance_metres=clearance, sample_spacing_metres=.05,
                dense_checked_positions=checks, closed=closed,
                path_length_metres=float(np.linalg.norm(edges, axis=1).sum()),
                maximum_frame_translation_metres=float(np.linalg.norm(edges, axis=1).max()),
                maximum_frame_rotation_degrees=float(angle.max()), orthonormal_error=error,
                minimum_foreground_fraction=min(foreground), mean_foreground_fraction=float(np.mean(foreground)),
                foreground_per_frame=foreground,
                collision_method='Float64 triangle/edge distances via complete-Mesh BVH; half-sample guard covers segments')
