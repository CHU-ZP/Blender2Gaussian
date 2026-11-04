"""Sample a continuous indoor route or an open-scene orbit from source Mesh geometry."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from mesh_to_gs.mesh_supervision import digest
from mesh_to_gs.trajectories import audit_trajectory, indoor_path, keyframe_indices, make_cameras, orbit_path


def draw_path(triangles, bounds, points, path, closed=False, footprint=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle
    varying_height = closed and np.ptp(points[:, 2]) > .01
    if varying_height:
        fig, (ax, height_ax) = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True,
                                          gridspec_kw={'width_ratios': [2, 1]})
        height_ax.plot(np.arange(len(points)), points[:, 2], color='#1473e6')
        height_ax.set(xlabel='Frame', ylabel='Camera height / m', title='Height along the orbit')
        height_ax.grid(alpha=.2)
    else:
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    if closed:
        low, high = np.asarray(bounds)
        ax.add_patch(Rectangle(low[:2], *(high-low)[:2], fill=False,
                               edgecolor='#929aa7', linestyle='--', label='Source bounds'))
        if footprint is None:
            stride = max(1, len(triangles)//50000)
            footprint = triangles[::stride].mean(axis=1)[:, :2]
        ax.scatter(footprint[:, 0], footprint[:, 1], color='#929aa7', s=.6, alpha=.2,
                   rasterized=True)
        segments = []
    else:
        segments = mesh_section(triangles, bounds, points)
    if segments:
        ax.add_collection(LineCollection(np.concatenate(segments), colors='#b0b5bd', linewidths=.4))
    route = np.r_[points, points[:1]] if closed else points
    ax.plot(route[:, 0], route[:, 1], color='#1473e6', linewidth=2.5)
    frames = keyframe_indices(len(points), closed)
    ax.scatter(points[frames, 0], points[frames, 1], c=np.arange(6), cmap='viridis', s=50, zorder=3)
    for i, frame in enumerate(frames):
        offset = (6, -15) if varying_height and i >= 3 else (6, 6)
        ax.annotate(str(i+1), points[frame, :2], xytext=offset, textcoords='offset points')
    if varying_height:
        height_ax.scatter(frames, points[frames, 2], c=np.arange(6), cmap='viridis', s=50)
        for i, frame in enumerate(frames):
            height_ax.annotate(str(i+1), (frame, points[frame, 2]), xytext=(4, 5), textcoords='offset points')
    ax.scatter(*points[0, :2], marker='>', color='#13a36d', s=110, label='Start', zorder=4)
    if not closed:
        ax.scatter(*points[-1, :2], marker='s', color='#d94f40', s=60, label='End', zorder=4)
    ax.set_aspect('equal')
    ax.autoscale()
    ax.margins(.08)
    ax.set(xlabel='World X / m', ylabel='World Y / m',
           title='Camera orbit and scene footprint' if closed else 'Camera trajectory and Mesh horizontal section')
    ax.legend()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def mesh_section(triangles, bounds, points):
    # Intersect Mesh faces with a horizontal plane for a readable floor plan.
    level = min(float(points[0, 2]), float(np.mean(np.asarray(bounds)[:, 2])))
    selected = triangles[(triangles[:, :, 2].min(1) <= level) & (triangles[:, :, 2].max(1) >= level)]
    segments = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        for c, d in ((0, 1), (1, 2), (2, 0)):
            if (a, b) >= (c, d):
                continue
            p, q, r, s = selected[:, a], selected[:, b], selected[:, c], selected[:, d]
            dz1, dz2 = q[:, 2]-p[:, 2], s[:, 2]-r[:, 2]
            t1 = np.divide(level-p[:, 2], dz1, out=np.full(len(p), np.nan), where=dz1 != 0)
            t2 = np.divide(level-r[:, 2], dz2, out=np.full(len(p), np.nan), where=dz2 != 0)
            keep = (t1 >= 0) & (t1 <= 1) & (t2 >= 0) & (t2 <= 1)
            segments.append(np.stack((p[keep, :2]+(q-p)[keep, :2]*t1[keep, None],
                                      r[keep, :2]+(s-r)[keep, :2]*t2[keep, None]), 1))
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=288)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--clearance', type=float, default=.25)
    parser.add_argument('--target', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Fixed indoor gaze target in world metres; default follows the route')
    parser.add_argument('--orbit-turns', type=int, default=1)
    parser.add_argument('--height-variation', type=float, default=0.,
                        help='Open-scene orbit height rise in metres, smoothly returning to its starting height')
    parser.add_argument('--gpu', type=int, help='CUDA device index used for the Mesh audit')
    args = parser.parse_args()
    if min(args.frames, args.fps, args.width, args.height) < 2 or args.clearance <= 0:
        parser.error('Positive clearance and counts of at least two are required')
    if args.orbit_turns < 1 or not np.isfinite(args.height_variation) or args.height_variation < 0:
        parser.error('Positive orbit turns and nonnegative finite height variation are required')
    if args.output_dir.exists():
        parser.error('Use a new trajectory directory')
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import bpy
    from mesh_to_gs.blend import extract_scene
    from mesh_to_gs.visibility import VisibilityTester
    report = json.loads((args.model_dir/'provenance/initializer.json').read_text())
    source = Path(report['source']).resolve()
    mode = report['visibility']['mode']
    if args.target is not None and (mode != 'indoor' or not np.isfinite(args.target).all()):
        parser.error('--target requires an indoor scene and finite world coordinates')
    if mode == 'indoor' and (args.orbit_turns != 1 or args.height_variation != 0):
        parser.error('Orbit turns and height variation require an open scene')
    tick = time.perf_counter()
    print(f'Extracting source geometry: {source}', flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False)
    groups, bounds, _ = extract_scene()
    triangles = np.concatenate([part for group in groups.values() for part in group['triangles']])
    del groups
    bpy.ops.wm.read_factory_settings(use_empty=True)
    print(f'Checking {len(triangles):,} Mesh triangles for a {mode} trajectory', flush=True)
    tester = VisibilityTester.for_scene(triangles, bounds, mode, spacing=.5, directions=48)
    if mode == 'indoor':
        points, sampling = indoor_path(tester, triangles, bounds, args.frames, args.seed,
                                      args.frames/args.fps*1.2, args.clearance)
        target, closed = args.target, False
    else:
        points, target, sampling = orbit_path(bounds, args.frames, args.seed, args.width, args.height,
                                             turns=args.orbit_turns, height_variation=args.height_variation)
        closed = True
    cameras = make_cameras(points, args.width, args.height, args.fps, target=target)
    sampling['gaze'] = dict(mode='follow_route' if target is None else 'fixed_target',
                            target=None if target is None else np.asarray(target).tolist())
    audit = audit_trajectory(tester, triangles, cameras, args.clearance, closed)
    args.output_dir.mkdir(parents=True)
    data = dict(width=args.width, height=args.height, convention='OpenCV', cameras=cameras,
                fps=args.fps, duration_seconds=args.frames/args.fps, closed=closed,
                trajectory_mode=mode, seed=args.seed)
    (args.output_dir/'cameras.json').write_text(json.dumps(data, indent=2)+'\n')
    draw_path(triangles, bounds, points, args.output_dir/'path.png', closed=closed)
    record = dict(status='pass', source_blend=str(source), source_sha256=digest(source),
                  model_sha256=digest(args.model_dir/'model.ply'),
                  cameras_sha256=digest(args.output_dir/'cameras.json'), source_triangles=len(triangles),
                  scene_bounds=bounds, sampling=sampling, audit=audit, seconds=time.perf_counter()-tick,
                  geometry_cache='Evaluated in memory; no Mesh cache or training output created')
    (args.output_dir/'trajectory_report.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'foreground_per_frame'}), flush=True)


if __name__ == '__main__':
    main()
