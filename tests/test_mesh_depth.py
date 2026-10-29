"""Verify the depth convention used by Mesh/Gaussian acceptance."""
import numpy as np


def test_cycles_z_is_camera_axis_depth_at_wide_fov(tmp_path):
    import bpy
    import OpenEXR
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    scene.render.engine='CYCLES'
    scene.cycles.device='CPU'
    scene.cycles.samples=1
    scene.cycles.use_denoising=False
    scene.render.resolution_x,scene.render.resolution_y,scene.render.resolution_percentage=16,12,100
    bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-3))
    bpy.ops.object.camera_add()
    scene.camera=bpy.context.object
    scene.camera.data.lens=20
    scene.view_layers[0].use_pass_z=True
    scene.render.image_settings.media_type='MULTI_LAYER_IMAGE'
    scene.render.image_settings.color_depth='32'
    path=tmp_path/'plane.exr'
    scene.render.filepath=str(path)
    bpy.ops.render.render(write_still=True)
    image=OpenEXR.File(str(path),separate_channels=True)
    depth=next(v.pixels for part in image.parts for k,v in part.channels.items() if k.endswith('.Depth.Z'))
    # Ray distance would increase towards the corners; camera-Z remains 3.
    np.testing.assert_allclose(depth,3,atol=1e-6)
