"""Compare Gaussian renders against matched opaque Mesh references, per camera group.

Metrics per view (Mesh foreground / interior as defined below):
  basecolor_psnr_db, basecolor_mae: sRGB unlit BaseColor on all Mesh foreground;
  opaque_coverage: Gaussian alpha >= 0.95 on the Mesh interior (eroded 3 px,
      excluding 3 px around Mesh depth contours);
  depth_consistent_coverage: also camera-Z error <= max(3 cm, 1% depth);
  depth_mae_world_units, depth_within_3cm: Mesh first hit vs Gaussian
      alpha-normalized expected depth (different estimators);
  contour_f1/precision/recall: depth jumps > max(5 cm, 2%) or mask edges, 2 px;
  spill: Mesh foreground where the Gaussian is opaque (alpha >= 0.5) and its
      expected depth is in front of the Mesh first hit by > max(5 cm, 1% depth).
      5 cm (3 cm before 2026-11-01): enlarged support on oblique planes shifts
      the expected depth ~3 cm towards the camera without crossing any edge.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

SPILL_DEPTH = .05
KEYS = ['basecolor_psnr_db', 'basecolor_mae', 'opaque_coverage', 'depth_consistent_coverage',
        'depth_mae_world_units', 'depth_within_3cm', 'contour_f1', 'contour_precision', 'contour_recall', 'spill',
        'silhouette_overflow', 'full_foreground_opaque_coverage', 'boundary_opaque_coverage',
        'full_foreground_alpha_min', 'boundary_alpha_mean', 'outside_alpha_above_05',
        'outside_alpha_max_beyond_edge_band']


def contours(depth, valid):
    edge = np.zeros(depth.shape, bool)
    for axis in (0, 1):
        first, second = [slice(None)] * 2, [slice(None)] * 2
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(first), tuple(second)
        jump = np.abs(depth[a] - depth[b]) > np.maximum(.05, .02 * np.minimum(depth[a], depth[b]))
        jump = (jump & valid[a] & valid[b]) | (valid[a] != valid[b])
        edge[a] |= jump
        edge[b] |= jump
    return edge


def contour_score(reference, predicted, tolerance=2):
    precision = np.sum(predicted & binary_dilation(reference, iterations=tolerance)) / max(predicted.sum(), 1)
    recall = np.sum(reference & binary_dilation(predicted, iterations=tolerance)) / max(reference.sum(), 1)
    return float(precision), float(recall), float(2 * precision * recall / max(precision + recall, 1e-12))


def view_metrics(reference, candidate, name, edge_band_pixels=0):
    md, ma = np.load(reference / f'{name}_depth.npy'), np.load(reference / f'{name}_alpha.npy')
    gd, ga = np.load(candidate / f'{name}_depth.npy'), np.load(candidate / f'{name}_alpha.npy')
    mesh = (ma >= .99) & (md > 0)
    edges = contours(md, mesh)
    interior = binary_erosion(mesh, iterations=3) & ~binary_dilation(edges, iterations=3)
    boundary = mesh & ~interior
    outside = (ma <= .01) & (distance_transform_edt(ma <= .01) > edge_band_pixels)
    opaque = ga >= .95
    error = np.abs(gd - md)
    rgbm = np.asarray(Image.open(reference / f'{name}.png').convert('RGB'), float) / 255
    rgbg = np.asarray(Image.open(candidate / f'{name}.png').convert('RGB'), float) / 255
    precision, recall, f1 = contour_score(edges, contours(gd, opaque & (gd > 0)))
    spill = mesh & (ga >= .5) & (gd < md - np.maximum(SPILL_DEPTH, .01 * md))
    def mean(value, mask):
        return float(value[mask].mean()) if mask.any() else None
    return dict(camera=name, interior_pixels=int(interior.sum()),
                basecolor_psnr_db=float(-10 * np.log10(max(np.mean((rgbm[mesh] - rgbg[mesh]) ** 2), 1e-12))) if mesh.any() else None,
                basecolor_mae=mean(np.abs(rgbm - rgbg), mesh),
                opaque_coverage=mean(opaque, interior),
                depth_consistent_coverage=mean(opaque & (error <= np.maximum(.03, .01 * md)), interior),
                depth_mae_world_units=mean(error, interior),
                depth_within_3cm=mean(error <= .03, interior),
                contour_f1=f1, contour_precision=precision, contour_recall=recall,
                spill=float(spill.sum() / max(mesh.sum(), 1)),
                silhouette_overflow=float(((ma <= .01) & (ga >= .5)).sum() / max(mesh.sum(), 1)),
                full_foreground_opaque_coverage=float(opaque[mesh].mean()) if mesh.any() else None,
                boundary_opaque_coverage=float(opaque[boundary].mean()) if boundary.any() else None,
                full_foreground_alpha_min=float(ga[mesh].min()) if mesh.any() else None,
                boundary_alpha_mean=float(ga[boundary].mean()) if boundary.any() else None,
                edge_band_pixels=edge_band_pixels,
                outside_alpha_above_05=float((outside & (ga > .05)).sum() / max(mesh.sum(), 1)),
                outside_alpha_max_beyond_edge_band=float(ga[outside].max()) if outside.any() else None)


def summarize(views, roles=None):
    groups = {'all': views}
    if roles:
        for role in sorted(set(roles.values())):
            groups[role] = [v for v in views if roles.get(v['camera']) == role]
    return {g: {k: float(np.mean([v[k] for v in vs if v[k] is not None]))
                if any(v[k] is not None for v in vs) else None for k in KEYS}
            for g, vs in groups.items() if vs}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--cameras', type=Path, required=True)
    parser.add_argument('--model', nargs=2, action='append', metavar=('LABEL', 'RENDERS'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--edge-tolerance-world', type=float, default=0.,
                        help='Declared silhouette error band, projected conservatively using nearest Mesh depth')
    args = parser.parse_args()
    cameras = json.loads(args.cameras.read_text())['cameras']
    roles = {c['name']: c.get('role', 'all') for c in cameras}
    result = {'reference': str(args.reference), 'cameras': str(args.cameras), 'models': {}}
    for label, renders in args.model:
        views = []
        for c in cameras:
            band = 0
            if args.edge_tolerance_world > 0:
                depth = np.load(args.reference / f'{c["name"]}_depth.npy')
                valid = depth[depth > 0]
                if len(valid):
                    K = np.asarray(c['K'])
                    image = Image.open(args.reference / f'{c["name"]}.png')
                    factor = 1 + np.hypot(image.width / K[0, 0], image.height / K[1, 1])
                    band = int(np.ceil(max(K[0, 0], K[1, 1]) * args.edge_tolerance_world * factor
                                       / max(float(valid.min()) - args.edge_tolerance_world, 1e-12)))
            views.append(view_metrics(args.reference, Path(renders), c['name'], band))
        result['models'][label] = {'renders': renders, 'mean': summarize(views, roles), 'views': views}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    header = f"{'model':34s} " + ' '.join(f'{k[:10]:>10s}' for k in KEYS)
    for group in next(iter(result['models'].values()))['mean']:
        print(f'[{group}]')
        print(header)
        for label, entry in result['models'].items():
            m = entry['mean'][group]
            print(f'{label:34s} ' + ' '.join(f'{m[k]:10.4f}' if m[k] is not None else f'{"n/a":>10s}' for k in KEYS))


if __name__ == '__main__':
    main()
