"""Generate geometry-selected training and validation cameras and PBR targets."""
import argparse
import json
from pathlib import Path

import numpy as np

from mesh_to_gs.mesh_supervision import digest, model_inputs
from mesh_to_gs.view_selection import candidate_views, greedy_views, surface_targets, view_features
from mesh_to_gs.visibility import VisibilityTester, mesh_near_surface


def select_cameras(model, views, validation_views, seed, width, height):
    from mesh_to_gs.cameras import camera_entry, look_at
    from mesh_to_gs.view_selection import distinct_pose

    report, metadata = model_inputs(model)
    bounds = np.asarray(report['source_bounds'], np.float64)
    origin = bounds.mean(0)
    # Use scene-relative units for camera placement, then restore world poses.
    unit = np.linalg.norm(bounds[1] - bounds[0]) / 16
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError('Invalid mesh bounds')
    triangles = (np.load(metadata/'triangles.npy') - origin) / unit
    groups = json.loads((metadata/'groups.json').read_text())
    half_extent = np.maximum((bounds[1]-origin)/unit, .5)
    tester = VisibilityTester.for_scene(triangles, [-half_extent, half_extent], spacing=.5, directions=32)
    rng = np.random.default_rng(seed)
    targets, owners = surface_targets(triangles, groups, 32768, rng)
    total = views + validation_views
    try:
        pool, placement = candidate_views(tester, triangles, targets, owners, [], 4*total, rng, width, height)
    except ValueError as exc:
        # Open assets and thin/floorless scenes may have no indoor viewpoints.
        pool = []
        for _ in range(32*total):
            direction = rng.normal(size=3)
            direction[2] *= .65
            eye = direction/np.linalg.norm(direction)*rng.uniform(8.5, 14.)
            target = targets[rng.integers(len(targets))] * .5
            entry = camera_entry('candidate', look_at(eye, target), width, height, 60.)
            if distinct_pose(entry, pool):
                pool.append(entry)
            if len(pool) >= 4*total:
                break
        placement = dict(fallback='exterior viewpoints', reason=str(exc))
    matrix, _, feature_keys, stats = view_features(tester, triangles, groups, pool)
    eyes = np.array([c['camera_to_world'] for c in pool])[:, :3, 3]
    usable = ~mesh_near_surface(tester, triangles, eyes, .3)
    usable &= np.array([s['foreground'] >= .05 for s in stats])
    eligible = np.flatnonzero(usable)
    if len(eligible) < total:
        raise ValueError(f'Only {len(eligible)} useful distinct cameras for {total} requested')
    validation = rng.choice(eligible, validation_views, replace=False)
    usable[validation] = False
    remaining = np.flatnonzero(usable)
    train_matrix = matrix[remaining]
    columns = np.unique(train_matrix.indices)
    feature_groups = feature_keys[columns, 0]
    weights = 1 / np.sqrt(np.bincount(feature_groups)[feature_groups])
    chosen, coverage = greedy_views(train_matrix[:, columns], weights, 0, views)
    train = remaining[chosen]
    def restore(indices, role):
        cameras = []
        for i, j in enumerate(indices):
            c = pool[j]
            pose = np.array(c['camera_to_world'])
            pose[:3, 3] = origin + unit*pose[:3, 3]
            cameras.append(camera_entry(f'{role}_{i:03d}', pose, width, height,
                                        c['horizontal_fov_degrees'], near=.05*unit, far=250*unit))
        return dict(width=width, height=height, convention='OpenCV', cameras=cameras)
    return restore(train, 'train'), restore(validation, 'validation'), dict(
        placement=placement, normalized_world_unit=unit, candidates=len(pool),
        eligible=len(eligible), training_coverage=coverage, validation_indices=validation.tolist())


def main():
    from mesh_to_gs.mesh_reference import render_references, MATERIAL_LIGHTING_PROFILE, MATERIAL_LIGHTING_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    for key in ('views', 'validation-views', 'seed', 'width', 'height', 'samples'):
        parser.add_argument('--'+key, type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report, _ = model_inputs(args.model)
    train, validation, selection = select_cameras(
        args.model, args.views, args.validation_views, args.seed, args.width, args.height)
    for name, data in [('training_cameras', train), ('validation_cameras', validation), ('selection', selection)]:
        (args.output/(name+'.json')).write_text(json.dumps(data, indent=2)+'\n')
    files = {}
    for camera, path, _, error, _ in render_references(
            Path(report['source']), train['cameras'] + validation['cameras'], args.width,
            args.height, args.output/'training_references', args.samples, material=True):
        files[camera['name']] = dict(sha256=digest(path), projection_error_px=error)
    (args.output/'refinement_report.json').write_text(json.dumps(dict(
        blend=report['source'], source_sha256=report['input_sha256'], reference_mode='material',
        reference_lighting=MATERIAL_LIGHTING_PROFILE, lighting_sha256=digest(MATERIAL_LIGHTING_CONFIG),
        samples=args.samples, views=files), indent=2)+'\n')


if __name__ == '__main__':
    main()
