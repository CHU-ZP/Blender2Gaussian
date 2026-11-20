import numpy as np
import pytest
from plyfile import PlyData

from mesh_to_gs.area_sampling import (allocate_area_counts, initialize_area_sampled,
                                      uniform_triangle_barycentrics)
from mesh_to_gs.color import linear_to_srgb
from mesh_to_gs.mesh_supervision import model_inputs


def test_area_allocation_preserves_budget_and_area_proportions():
    areas = np.array([1., 3., 6., .25])
    counts = allocate_area_counts(areas, 100_000)
    assert counts.sum() == 100_000
    np.testing.assert_allclose(counts/counts.sum(), areas/areas.sum(), atol=1e-5)
    assert np.all(counts >= 0)


def test_spacing_counts_follow_area_independently_of_tessellation():
    # Fractional allocations survive splitting a surface into many small faces.
    np.testing.assert_array_equal(allocate_area_counts([.25, .75], spacing=.25), [4, 12])
    assert allocate_area_counts(np.full(128, 1/128), spacing=.25).sum() == 16
    assert allocate_area_counts([.25, .75], spacing=.125).sum() == 64
    np.testing.assert_array_equal(allocate_area_counts([.001, .01], spacing=1.), [0, 1])
    for invalid in (0, -1., np.nan, np.inf):
        with pytest.raises(ValueError, match='spacing'):
            allocate_area_counts([1.], spacing=invalid)
    with pytest.raises(ValueError, match='without total'):
        allocate_area_counts([1.], 100, spacing=.1)


def test_barycentric_draws_are_uniform_inside_triangle_and_repeatable():
    first = uniform_triangle_barycentrics(np.random.default_rng(17), 100_000)
    second = uniform_triangle_barycentrics(np.random.default_rng(17), 100_000)
    np.testing.assert_array_equal(first, second)
    assert first.min() >= 0
    np.testing.assert_allclose(first.sum(1), 1., atol=1e-14)
    np.testing.assert_allclose(first.mean(0), np.full(3, 1/3), atol=.002)


def test_area_sampled_model_has_mesh_training_schema_and_atomic_publish(tmp_path):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new('area fixture')
    mesh.from_pydata([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [], [[0, 1, 2]])
    obj = bpy.data.objects.new('area fixture', mesh)
    bpy.context.collection.objects.link(obj)
    material = bpy.data.materials.new('blue'); material.use_nodes = True
    material.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = (.05, .2, .8, 1.)
    mesh.materials.append(material)
    source = tmp_path/'area.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(source))

    output = tmp_path/'sampled'
    report = initialize_area_sampled(source, output, total=2048, seed=8)
    assert report['gaussians'] == 2048 and (output/'initialization_report.json').is_file()
    _, metadata = model_inputs(output)
    for name in ('triangles.npy', 'uv.npy', 'chart.npy', 'groups.json', 'face.npy',
                 'barycentric.npy', 'boundary.npy'):
        assert (metadata/name).is_file()
    rows = PlyData.read(output/'model.ply')['vertex'].data
    faces = np.load(metadata/'face.npy', mmap_mode='r')
    bary = np.load(metadata/'barycentric.npy', mmap_mode='r')
    np.testing.assert_array_equal(faces, np.zeros(2048, np.int32))
    assert np.all(bary >= 0) and np.all(bary.sum(1) <= 1)
    rgb = np.c_[rows['f_dc_0'], rows['f_dc_1'], rows['f_dc_2']]*.28209479177387814+.5
    expected = linear_to_srgb(np.asarray([.05, .2, .8], np.float32))
    np.testing.assert_allclose(rgb, np.broadcast_to(expected, rgb.shape), atol=1e-6)
    with pytest.raises(FileExistsError):
        initialize_area_sampled(source, output, total=2048, seed=8)

    spaced = tmp_path/'spaced'
    report = initialize_area_sampled(source, spaced, spacing=.25, seed=8)
    assert report['gaussians'] == 8
    assert report['target_spacing_world_units'] == .25
    assert report['tangent_sigma_world_units'] == pytest.approx(.2)
    assert model_inputs(spaced)[0]['method'] == 'mesh_area_spacing_barycentric_texture'
    assert len(PlyData.read(spaced/'model.ply')['vertex'].data) == 8

