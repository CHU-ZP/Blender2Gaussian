"""Unlit opaque Mesh BaseColor references (Cycles) at OpenCV-convention cameras."""
from pathlib import Path
import time

import bpy
from mathutils import Matrix
import numpy as np
import OpenEXR


def configure_device(scene):
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for backend in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            selected = []
            for device in prefs.devices:
                device.use = device.type == backend
                if device.use:
                    selected.append(device.name)
            if selected:
                scene.cycles.device = 'GPU'
                return {'backend': backend, 'devices': selected}
        except (TypeError, RuntimeError):
            continue
    scene.cycles.device = 'CPU'
    return {'backend': 'CPU', 'devices': ['CPU']}


def configure_camera(scene, camera, entry, width, height):
    K = np.array(entry['K'], dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    camera.matrix_world = Matrix((np.array(entry['camera_to_world']) @ np.diag([1, -1, -1, 1])).tolist())
    scene.render.pixel_aspect_x = 1
    scene.render.pixel_aspect_y = fx / fy
    data = camera.data
    data.type, data.sensor_fit, data.sensor_width = 'PERSP', 'HORIZONTAL', 36
    data.lens = fx * data.sensor_width / width
    data.shift_x = (width / 2 - cx) / width
    data.shift_y = (cy - height / 2) * (fx / fy) / width
    data.clip_start, data.clip_end = entry['near'], entry['far']
    data.dof.use_dof = False
    bpy.context.view_layer.update()
    # Check the Blender projection against the saved OpenCV intrinsics.
    projection = np.array(camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(),
                          x=width, y=height, scale_x=1, scale_y=fx / fy))
    cv = np.array([[0, 0, 3, 1], [.5, .3, 3, 1], [-.6, -.4, 4, 1]], dtype=float)
    clip = cv @ np.diag([1, -1, -1, 1]) @ projection.T
    ndc = clip[:, :2] / clip[:, 3:4]
    pixels = np.column_stack(((ndc[:, 0] + 1) * width / 2, (1 - ndc[:, 1]) * height / 2))
    expected = cv[:, :3] @ K.T
    expected = expected[:, :2] / expected[:, 2:3]
    error = float(np.max(np.abs(pixels - expected)))
    if error > .001:
        raise ValueError(f'Camera projection mismatch: {error} pixels')
    return error


def base_color_materials():
    """Route every Principled Base Color into a unit-strength emission."""
    for material in bpy.data.materials:
        if material.node_tree is None:
            continue
        tree = material.node_tree
        outputs = [node for node in tree.nodes if node.type == 'OUTPUT_MATERIAL' and node.is_active_output]
        if len(outputs) != 1 or len(outputs[0].inputs['Surface'].links) != 1:
            raise ValueError(f'Unsupported output: {material.name}')
        bsdf = outputs[0].inputs['Surface'].links[0].from_node
        if bsdf.type != 'BSDF_PRINCIPLED':
            raise ValueError(f'Expected Principled BSDF: {material.name}')
        emission = tree.nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = bsdf.inputs['Base Color'].default_value
        emission.inputs['Strength'].default_value = 1
        if bsdf.inputs['Base Color'].is_linked:
            tree.links.new(bsdf.inputs['Base Color'].links[0].from_socket, emission.inputs['Color'])
        tree.links.new(emission.outputs[0], outputs[0].inputs['Surface'])


def read_exr(path):
    image = OpenEXR.File(str(path), separate_channels=True)
    channels = {k: v.pixels for part in image.parts for k, v in part.channels.items()}
    def channel(suffix):
        return next(v for k, v in channels.items() if k.endswith(suffix))
    rgb = np.stack([channel('.Combined.' + c) for c in 'RGB'], axis=-1)
    return rgb, channel('.Combined.A'), channel('.Depth.Z')


def load_blend(blend):
    bpy.ops.wm.open_mainfile(filepath=str(Path(blend).resolve()), load_ui=False)


def render_references(blend, cameras, width, height, output_dir, samples=16):
    """Render each camera to <name>.exr (premultiplied scene-linear RGB, alpha, camera-Z).

    Yields (camera entry, exr path, seconds, projection error in pixels).
    """
    load_blend(blend)
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    device = configure_device(s)
    s.render.resolution_x, s.render.resolution_y = width, height
    s.render.resolution_percentage = 100
    s.render.film_transparent = True
    s.render.use_compositing = s.render.use_sequencer = False
    s.render.use_persistent_data = True
    # Blender 5.2 selects multi-part EXR through media_type.
    s.render.image_settings.media_type = 'MULTI_LAYER_IMAGE'
    s.render.image_settings.color_depth = '32'
    s.render.image_settings.exr_codec = 'ZIP'
    s.view_layers[0].use_pass_z = True
    s.cycles.samples = samples
    s.cycles.use_denoising = s.cycles.use_adaptive_sampling = False
    s.cycles.seed = 20261029
    s.view_settings.view_transform, s.view_settings.look = 'Standard', 'None'
    s.view_settings.exposure, s.view_settings.gamma = 0, 1
    base_color_materials()
    data = bpy.data.cameras.new('Geometry reference camera')
    cam = bpy.data.objects.new(data.name, data)
    s.collection.objects.link(cam)
    s.camera = cam
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for entry in cameras:
        error = configure_camera(s, cam, entry, width, height)
        path = Path(output_dir) / (entry['name'] + '.exr')
        s.render.filepath = str(path.resolve())
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        yield entry, path, time.perf_counter() - start, error, device
