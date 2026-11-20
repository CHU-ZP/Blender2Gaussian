"""Cycles Mesh references with BaseColor or illuminated Principled materials."""
from pathlib import Path
import hashlib
import json
import time

import bpy
from mathutils import Matrix, Vector
import numpy as np
import OpenEXR
from mesh_to_gs.color import MATERIAL_EXPOSURE_STOPS


# Cycles EXR Combined is scene-linear; callers apply the shared fixed exposure
# and display transform to the saved radiance before sRGB encoding.
MATERIAL_LIGHTING_PROFILE = 'uniform_interior_softbox_grid_v10'
MATERIAL_DISPLAY_TRANSFORM = 'aces_fitted'
MATERIAL_INTERIOR_SOFTBOX_ENERGY = 5.0  # Base downward panel power; other groups are recorded separately.
MATERIAL_INTERIOR_SOFTBOX_COUNT = 68  # 24 downward + 24 upward + 20 side panels for sk_00918.
MATERIAL_LIGHTING_CONFIG = Path(__file__).resolve().parents[1]/'configs/material_lighting_v10.json'


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


def _scene_bounds():
    """Return world-space bounds for placing deterministic auxiliary lights."""
    corners = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH' or not obj.bound_box:
            continue
        corners.extend(np.asarray(obj.matrix_world @ Vector(c), dtype=np.float64) for c in obj.bound_box)
    if not corners:
        return np.array([-1., -1., -1.]), np.array([1., 1., 1.])
    points = np.asarray(corners, dtype=np.float64)
    return points.min(0), points.max(0)


def _point_light_at(name, kind, location, target, *, energy, size=1.):
    """Create a temporary Area/Sun light aimed at a world-space target."""
    data = bpy.data.lights.new(name, type=kind)
    data.energy = float(energy)
    if kind == 'AREA':
        data.shape = 'DISK'
        data.size = float(size)
    obj = bpy.data.objects.new(name, data)
    obj['mesh_reference_generated'] = True
    bpy.context.scene.collection.objects.link(obj)
    obj.location = Vector(np.asarray(location, dtype=np.float64))
    direction = Vector(np.asarray(target, dtype=np.float64))-obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    return obj


def _setup_legacy_material_lighting():
    """Add a stable studio light rig for source scenes without light objects.

    The retained sk_00918 blend has no LIGHT objects.  Material rendering
    without illumination produces nearly black references, so material mode
    adds temporary lights after loading the blend.  The rig is scene-relative
    and is never written back to the source blend.
    """
    scene = bpy.context.scene
    lower, upper = _scene_bounds()
    center = (lower+upper)/2
    extent = np.maximum(upper-lower, 1.)
    radius = float(np.linalg.norm(extent))
    # A large-area light dome makes illumination stable across training and
    # test cameras.  The panels are deliberately much larger than the room,
    # so brightness changes slowly over a surface instead of producing small
    # hot spots tied to one camera.  The broad exterior panels provide soft
    # window-like directionality, while the interior fill below prevents the
    # shelves and furniture from falling into a black cavity.
    panel_size = max(12., 1.8*radius)
    panel_specs = [
        ('MeshRef_CeilingPanel', np.array([0., 0., 2.8]), 2600.),
        ('MeshRef_FrontPanel', np.array([0., -2.4, .9]), 1560.),
        ('MeshRef_BackPanel', np.array([0., 2.4, .9]), 1560.),
        ('MeshRef_LeftPanel', np.array([-2.4, 0., .9]), 1560.),
        ('MeshRef_RightPanel', np.array([2.4, 0., .9]), 1560.),
    ]
    lights = [
        _point_light_at(name, 'AREA', center+direction*extent, center,
                        energy=energy, size=panel_size)
        for name, direction, energy in panel_specs
    ]
    # Four broad overhead corner panels provide cross-fill without introducing
    # the hard directional shadowing of a key light.  Their lower power keeps
    # the ceiling and floor from becoming a visible hotspot.
    for sx in (-1., 1.):
        for sy in (-1., 1.):
            lights.append(_point_light_at(
                f'MeshRef_Diagonal_{"L" if sx < 0 else "R"}{"F" if sy < 0 else "B"}',
                'AREA', center+np.array([1.8*sx, 1.8*sy, 1.3])*extent,
                center, energy=200., size=panel_size))
    # The scene is an interior room. Exterior panels cannot illuminate the
    # shelves and furniture behind the walls, so add broad ceiling softboxes.
    # Avoid lights on the room boundary: training cameras can approach those
    # surfaces, and a nearby emitter clips the whole view even at low power.
    interior_size = max(8., .85*float(min(extent[0], extent[1])))
    ceiling_z = float(upper[2] - .25*max(extent[2], .5))
    interior_positions = (-.15, .15)
    for ix, sx in enumerate(interior_positions):
        for iy, sy in enumerate(interior_positions):
            lights.append(_point_light_at(
                f'MeshRef_InteriorCeiling_{ix}{iy}', 'AREA',
                center+np.array([sx*extent[0], sy*extent[1], ceiling_z-center[2]]),
                center, energy=24., size=interior_size))
    # Keep a very weak sun only for numerical support on grazing surfaces; the
    # broad panels and world fill carry the actual appearance target.
    lights.append(_point_light_at('MeshRef_Sun', 'SUN', center+np.array([.4, -.3, 1.5])*extent,
                                  center, energy=.2))
    # Keep a small ambient term while retaining a transparent black camera
    # background.  Existing world nodes are preserved except for this
    # deterministic background strength/color used for material illumination.
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new('MeshRefWorld')
        scene.world = world
    world.use_nodes = True
    background = next((n for n in world.node_tree.nodes if n.type == 'BACKGROUND'), None)
    if background is None:
        background = world.node_tree.nodes.new('ShaderNodeBackground')
    # A neutral world fill reaches the recessed shelves and room corners that
    # no finite panel can see directly.  It is deliberately below the panel
    # contribution so the broad directional shading and material texture stay
    # visible instead of turning into a flat albedo pass.
    background.inputs['Color'].default_value = (0.35, 0.35, 0.35, 1.)
    background.inputs['Strength'].default_value = 0.45
    return dict(mode='uniform_area_dome_interior_fill_v9', lights=[obj.name for obj in lights],
                bounds_lower=lower.tolist(), bounds_upper=upper.tolist(),
                world_color=[.35, .35, .35], world_strength=.45, sun_energy=.2,
                panel_size=panel_size, interior_panel_size=interior_size,
                interior_softbox_count=4,
                interior_softbox_energy=24.,
                panel_count=len(lights),
                panel_energy_scale='large-area dome; reduced exterior five primary + four diagonal + four interior ceiling fills')


def setup_material_lighting():
    """Distributed interior softboxes, retaining the v9 exterior/world fill.

    The grid covers the room rather than concentrating four huge weak panels
    near its center. Upward panels sit farther below the ceiling to spread
    bounce illumination; open edges receive extra fill. All powers are fixed
    across cameras, and the source blend is never overwritten.
    """
    config = json.loads(MATERIAL_LIGHTING_CONFIG.read_text())
    if config['name'] != MATERIAL_LIGHTING_PROFILE:
        raise ValueError('Lighting config/profile mismatch')
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT' and obj.get('mesh_reference_generated'):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.lights.remove(data)
    legacy = _setup_legacy_material_lighting()
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT' and obj.get('mesh_reference_generated') and obj.name.startswith('MeshRef_InteriorCeiling'):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.lights.remove(data)
    scene = bpy.context.scene
    lower, upper = _scene_bounds()
    center = (lower+upper)/2
    deps = bpy.context.evaluated_depsgraph_get()
    xs = np.linspace(lower[0]+config['inset'], upper[0]-config['inset'], config['nx'])
    ys = np.linspace(lower[1]+config['inset'], upper[1]-config['inset'], config['ny'])
    placed, rejected = [], []

    def panel(name, location, target, energy, size, group):
        obj = _point_light_at(name, 'AREA', location, target, energy=energy, size=size)
        obj.visible_camera = False
        placed.append(dict(name=name, location=list(obj.location), target=list(target),
                           energy=energy, size=size, group=group))

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            hit, location, *_ = scene.ray_cast(deps, Vector((x, y, lower[2]+1.9)), Vector((0, 0, 1)))
            if hit and location.z < lower[2]+2.35:
                rejected.append(dict(x=float(x), y=float(y), reason='insufficient overhead clearance'))
                continue
            if not hit and not config['open_edge_fill']:
                rejected.append(dict(x=float(x), y=float(y), reason='no ceiling'))
                continue
            ceiling = float(location.z) if hit else float(upper[2])-.4
            z = min(ceiling-.25, float(upper[2])-.35)
            energy_scale = 1. if hit else config['open_edge_energy_scale']
            for side, energy, dz in [('Down', config['down_energy'], -2.), ('Up', config['up_energy'], 2.)]:
                light_z = z-config['up_lowering'] if side == 'Up' else z
                panel(f'MeshRef_Grid_{ix}_{iy}_{side}', [x, y, light_z], [x, y, light_z+dz],
                      energy*energy_scale, config['panel_size'], 'ceiling_'+side.lower())
    for axis in (0, 1):
        other = 1-axis
        positions = np.linspace(lower[other]+config['inset'], upper[other]-config['inset'],
                                config['nx'] if other == 0 else config['ny'])
        for side in (0, 1):
            for i, value in enumerate(positions):
                pos = center.copy()
                pos[axis] = lower[axis]+config['inset'] if side == 0 else upper[axis]-config['inset']
                pos[other], pos[2] = value, lower[2]+1.9
                target = pos.copy()
                target[axis] += 1. if side == 0 else -1.
                panel(f'MeshRef_WallWash_{axis}_{side}_{i}', pos, target,
                      config['wall_energy'], config['wall_size'], 'wall')
    bpy.context.view_layer.update()
    lights = [o.name for o in bpy.data.objects if o.type == 'LIGHT' and o.get('mesh_reference_generated')]
    return dict(mode=MATERIAL_LIGHTING_PROFILE, config=config,
                config_sha256=hashlib.sha256(MATERIAL_LIGHTING_CONFIG.read_bytes()).hexdigest(),
                bounds_lower=lower.tolist(), bounds_upper=upper.tolist(), lights=lights,
                panel_count=len(lights), interior_softbox_count=len(placed),
                interior_softbox_energy=config['down_energy'], interior_lights=placed,
                rejected_ceiling_cells=rejected,
                exterior=dict(world_color=legacy['world_color'], world_strength=legacy['world_strength'],
                              sun_energy=legacy['sun_energy'], panel_size=legacy['panel_size'],
                              profile='v9 exterior/world settings retained'))


def read_exr(path):
    image = OpenEXR.File(str(path), separate_channels=True)
    channels = {k: v.pixels for part in image.parts for k, v in part.channels.items()}
    def channel(suffix):
        return next(v for k, v in channels.items() if k.endswith(suffix))
    rgb = np.stack([channel('.Combined.' + c) for c in 'RGB'], axis=-1)
    return rgb, channel('.Combined.A'), channel('.Depth.Z')


def load_blend(blend):
    bpy.ops.wm.open_mainfile(filepath=str(Path(blend).resolve()), load_ui=False)


def render_references(blend, cameras, width, height, output_dir, samples=16, material=False,
                      material_lighting=True):
    """Render each camera to <name>.exr (scene-linear RGB, alpha, camera-Z).

    ``material=False`` preserves the historical emission BaseColor target.
    With ``material=True`` the original Principled materials are preserved.
    By default a fixed auxiliary rig and world fill generate the PBR target;
    ``material_lighting=False`` keeps the authored illumination instead.

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
    # Emission references are noise-free and keep the historical settings.
    # Material mode uses indirect illumination from the added rig; denoising
    # and adaptive sampling are required for a usable RGB target.
    s.cycles.use_denoising = bool(material)
    s.cycles.use_adaptive_sampling = bool(material)
    s.cycles.seed = 20261029
    s.view_settings.view_transform, s.view_settings.look = 'Standard', 'None'
    # Keep the recorded display exposure fixed while varying physical lights.
    # EXR Combined remains scene-linear; PNG/target conversion applies the
    # exposure once outside Blender. BaseColor keeps neutral display exposure.
    s.view_settings.exposure, s.view_settings.gamma = (MATERIAL_EXPOSURE_STOPS if material else 0), 1
    lighting = None
    if not material:
        base_color_materials()
    elif material_lighting:
        lighting = setup_material_lighting()
    data = bpy.data.cameras.new('Geometry reference camera')
    cam = bpy.data.objects.new(data.name, data)
    s.collection.objects.link(cam)
    s.camera = cam
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if lighting is not None:
        (Path(output_dir)/'lighting_report.json').write_text(json.dumps(lighting, indent=2)+'\n')
    for entry in cameras:
        error = configure_camera(s, cam, entry, width, height)
        path = Path(output_dir) / (entry['name'] + '.exr')
        s.render.filepath = str(path.resolve())
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        yield entry, path, time.perf_counter() - start, error, device
