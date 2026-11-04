"""OpenCV-convention pinhole cameras shared by camera exporters and joint refinement."""
import math

import numpy as np


def look_at(eye, target):
    """Camera-to-world matrix: +X right, +Y down, +Z forward, world +Z up."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    matrix[:3, 3] = eye
    return matrix


def camera_entry(name, camera_to_world, width, height, fov_degrees, near=.05, far=250., **extra):
    fx = width / (2 * math.tan(math.radians(fov_degrees) / 2))
    return dict(name=name, camera_to_world=np.asarray(camera_to_world).tolist(),
                view_matrix=np.linalg.inv(camera_to_world).tolist(),
                K=[[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], near=near, far=far,
                horizontal_fov_degrees=fov_degrees, **extra)


def pixel_rays(camera, columns, rows):
    """World-space unit directions through a regular grid of pixel centres."""
    width, height = 2 * camera['K'][0][2], 2 * camera['K'][1][2]
    u, v = np.meshgrid((np.arange(columns) + .5) * width / columns, (np.arange(rows) + .5) * height / rows)
    local = np.column_stack((u.ravel(), v.ravel(), np.ones(u.size))) @ np.linalg.inv(camera['K']).T
    world = local @ np.asarray(camera['camera_to_world'])[:3, :3].T
    return world / np.linalg.norm(world, axis=1)[:, None]


def training_cameras(tester, points, count, seed, width=1280, height=720, fov=70., distance=(1., 8.),
                     min_clearance=.3, min_foreground=.3, attempts=50):
    """Free-space cameras, each looking at a random sample visible from its eye.

    Eyes are visibility viewpoints with at least min_clearance to any surface;
    targets are model centres within `distance` whose line of sight is clear.
    Views with less than min_foreground geometry (coarse ray grid) are skipped.
    """
    rng = np.random.default_rng(seed)
    eyes = tester.viewpoints[tester.viewpoint_clearance >= min_clearance]
    if not len(eyes):
        raise ValueError('No free-space viewpoint has enough clearance for training views')
    cameras = []
    for _ in range(count * attempts):
        if len(cameras) == count:
            break
        eye = eyes[rng.integers(len(eyes))]
        target = points[rng.integers(len(points), size=256)]
        offset = target - eye
        length = np.linalg.norm(offset, axis=1)
        direction = offset / np.maximum(length, 1e-9)[:, None]
        usable = (length >= distance[0]) & (length <= distance[1]) & (np.abs(direction[:, 2]) < .95)
        if not usable.any():
            continue
        pick = np.flatnonzero(usable)
        clear = tester.first_hit(np.repeat(eye[None], len(pick), 0), direction[pick]) >= length[pick] - .02
        if not clear.any():
            continue
        target = target[pick[rng.choice(np.flatnonzero(clear))]]
        entry = camera_entry(f'train_{len(cameras):03d}', look_at(eye, target), width, height, fov)
        rays = pixel_rays(entry, 32, 18)
        foreground = np.isfinite(tester.first_hit(np.repeat(eye[None], len(rays), 0), rays)).mean()
        if foreground >= min_foreground:
            entry['foreground_estimate'] = float(foreground)
            cameras.append(entry)
    if len(cameras) < count:
        raise ValueError(f'Only {len(cameras)} of {count} training views found')
    return cameras

