"""Unlit opaque Mesh BaseColor, camera-Z and alpha at the Gaussian cameras."""
import argparse
import json
from pathlib import Path
import time

import bpy
import numpy as np
import OpenEXR
from PIL import Image

from render_mesh_reference import configure_device, configure_camera, base_color_materials


def read_exr(path):
    image = OpenEXR.File(str(path), separate_channels=True)
    channels = {k: v.pixels for part in image.parts for k, v in part.channels.items()}
    def channel(suffix):
        return next(v for k, v in channels.items() if k.endswith(suffix))
    rgb = np.stack([channel('.Combined.' + c) for c in 'RGB'], axis=-1)
    return rgb, channel('.Combined.A'), channel('.Depth.Z')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blend', type=Path, default=Path('data/sk_00918.blend'))
    p.add_argument('--cameras', type=Path, default=Path('outputs/sk_00918_validation/renders/cameras.json'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/sk_00918_geometry_reference'))
    p.add_argument('--samples', type=int, default=16)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cameras = json.loads(args.cameras.read_text())
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()), load_ui=False)
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    device = configure_device(s)
    s.render.resolution_x, s.render.resolution_y = cameras['width'], cameras['height']
    s.render.resolution_percentage = 100
    s.render.film_transparent = True
    s.render.use_compositing = s.render.use_sequencer = False
    s.render.use_persistent_data = True
    # Blender 5.2 selects multi-part EXR through media_type.
    s.render.image_settings.media_type = 'MULTI_LAYER_IMAGE'
    s.render.image_settings.color_depth = '32'
    s.render.image_settings.exr_codec = 'ZIP'
    s.view_layers[0].use_pass_z = True
    s.cycles.samples = args.samples
    s.cycles.use_denoising = s.cycles.use_adaptive_sampling = False
    s.cycles.seed = 20261029
    s.view_settings.view_transform, s.view_settings.look = 'Standard', 'None'
    s.view_settings.exposure, s.view_settings.gamma = 0, 1
    base_color_materials()
    data = bpy.data.cameras.new('Geometry reference camera')
    cam = bpy.data.objects.new(data.name, data)
    s.collection.objects.link(cam)
    s.camera = cam
    results = []
    for entry in cameras['cameras']:
        err = configure_camera(s, cam, entry, cameras['width'], cameras['height'])
        path = args.output_dir / (entry['name'] + '.exr')
        s.render.filepath = str(path.resolve())
        start = time.perf_counter()
        print('Rendering unlit geometry reference:', entry['name'], flush=True)
        bpy.ops.render.render(write_still=True)
        rgb, alpha, depth = read_exr(path)
        # Film is premultiplied scene-linear, composite white then encode sRGB.
        rgb = rgb + 1 - alpha[..., None]
        rgb = np.where(rgb <= .0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1/2.4) - .055)
        Image.fromarray((rgb.clip(0, 1)*255).round().astype('uint8')).save(path.with_suffix('.png'))
        # Cycles Z pass is camera-axis depth, verified with a wide-FOV plane at Z=3.
        valid = np.isfinite(depth) & (depth > entry['near']) & (depth < entry['far'])
        depth = np.where(valid, depth, 0).astype('float32')
        np.save(args.output_dir / (entry['name'] + '_depth.npy'), depth)
        np.save(args.output_dir / (entry['name'] + '_alpha.npy'), alpha)
        results.append(dict(camera=entry['name'], seconds=time.perf_counter()-start,
                            projection_error_pixels=err, valid_depth_fraction=float(valid.mean())))
        print(results[-1], flush=True)
    report = dict(blend=str(args.blend), cameras=str(args.cameras), device=device, samples=args.samples,
                  color='Principled BaseColor via unit emission; opaque, unlit, Standard sRGB',
                  depth='Cycles first-hit camera Z, world units; 0 denotes missing', source_saved=False, views=results)
    (args.output_dir / 'reference_report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
