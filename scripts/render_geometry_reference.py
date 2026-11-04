"""Unlit opaque Mesh BaseColor, camera-Z and alpha at the Gaussian cameras."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mesh_to_gs.mesh_reference import read_exr, render_references
from mesh_to_gs.mesh_supervision import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blend', type=Path, default=Path('data/sk_00918.blend'))
    p.add_argument('--cameras', type=Path, default=Path('configs/sk_00918_cameras_test.json'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/best/sk_00918/reference_test'))
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--rgb-only', action='store_true',
                   help='Keep RGB PNG only and remove each temporary EXR after conversion')
    args = p.parse_args()
    cameras = json.loads(args.cameras.read_text())
    results = []
    for entry, path, seconds, error, device in render_references(
            args.blend, cameras['cameras'], cameras['width'], cameras['height'], args.output_dir, args.samples):
        rgb, alpha, depth = read_exr(path)
        # Film is premultiplied scene-linear, composite white then encode sRGB.
        rgb = rgb + 1 - alpha[..., None]
        rgb = np.where(rgb <= .0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1/2.4) - .055)
        Image.fromarray((rgb.clip(0, 1)*255).round().astype('uint8')).save(path.with_suffix('.png'))
        # Cycles Z pass is camera-axis depth, verified with a wide-FOV plane at Z=3.
        valid = np.isfinite(depth) & (depth > entry['near']) & (depth < entry['far'])
        if args.rgb_only:
            path.unlink()
        else:
            np.save(args.output_dir / (entry['name'] + '_depth.npy'), np.where(valid, depth, 0).astype('float32'))
            np.save(args.output_dir / (entry['name'] + '_alpha.npy'), alpha)
        results.append(dict(camera=entry['name'], seconds=seconds, projection_error_pixels=error,
                            valid_depth_fraction=float(valid.mean())))
        print(results[-1], flush=True)
    report = dict(blend=str(args.blend), source_sha256=digest(args.blend),
                  cameras=str(args.cameras), cameras_sha256=digest(args.cameras),
                  width=cameras['width'], height=cameras['height'], device=device, samples=args.samples,
                  color='Principled BaseColor via unit emission; opaque, unlit, Standard sRGB',
                  depth=None if args.rgb_only else 'Cycles first-hit camera Z, world units; 0 denotes missing',
                  rgb_only=args.rgb_only, source_saved=False, views=results)
    (args.output_dir / 'reference_report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
