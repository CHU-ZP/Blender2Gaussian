"""Check supervision gradients, geometry labels and fixed parameter controls."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from mesh_to_gs.mesh_supervision import (local_queries, potential_candidates,
                                        sample_surface, surface_distribution)










def test_unit_local_coordinates_keep_precision_at_large_world_offsets():
    means = np.array([[100., 130., 1.]])
    p = means + np.array([[1e-5, 2e-5, 0.]])
    ids, local = local_queries(p, means, np.array([[1., 0, 0, 0]]),
                               np.array([[1e-5, 2e-5, 1e-7]]), cKDTree(means), 64, 1e-4)
    np.testing.assert_array_equal(ids, [[0]])
    np.testing.assert_allclose(local, [[[1., 1., 0.]]], atol=1e-6)


def test_candidate_selection_keeps_broad_support_behind_near_tiny_splats():
    means = np.array([[.001, 0, 0], [.002, 0, 0], [.05, 0, 0]])
    scales = np.array([[1e-6, 1e-6, 2e-8], [1e-6, 1e-6, 2e-8], [.1, .1, .002]])
    ids, _ = potential_candidates(np.zeros((1, 3)), means, np.tile([1., 0, 0, 0], (3, 1)),
                                   scales, cKDTree(means), 1, 1e-4)
    assert ids[0, 0] == 2


def test_surface_sampling_includes_unsampled_faces_and_excludes_unrepresented_charts():
    tri = np.array([[[0., 0, 0], [1., 0, 0], [0., 1., 0]],
                    [[1., 0, 0], [1., 1., 0], [0., 1., 0]],
                    [[2., 0, 0], [3., 0, 0], [2., 1., 0]]])
    probability = surface_distribution(tri, np.array([0, 0, 1]), np.array([0]))
    np.testing.assert_allclose(probability, [.75, .25, 0])
    points, faces, bary = sample_surface(tri, probability, 10000, np.random.default_rng(7))
    assert set(faces) == {0, 1} and (bary >= 0).all()
    np.testing.assert_allclose(bary.sum(1), 1)
    np.testing.assert_allclose(points[:, 2], 0)
    retained = surface_distribution(tri, np.array([0, 0, 1]), np.array([0]), retain_all=True)
    assert (retained > 0).all()
    np.testing.assert_allclose(retained, [2/3, 1/6, 1/6])
    area = surface_distribution(tri, np.array([0, 0, 1]), np.array([0]), retain_all=True, mode='area')
    other = surface_distribution(tri, np.array([0, 0, 1]), np.array([0, 1, 1, 2]), retain_all=True, mode='area')
    np.testing.assert_array_equal(area, other)
    np.testing.assert_allclose(area, [1/3] * 3)


def test_unsigned_distance_considers_another_nearby_open_sheet():
    import warp as wp
    from mesh_to_gs.visibility import mesh_distance
    wp.config.quiet = True
    wp.init()
    origin = np.array([100., 130., 1.])
    tri = np.array([[0., 0, 0], [1., 0, 0], [0., 1., 0],
                    [0., 0, .002], [1., 0, .002], [0., 1., .002]], dtype=np.float32)
    mesh = wp.Mesh(points=wp.array(tri, dtype=wp.vec3, device='cpu'),
                   indices=wp.array(np.arange(6, dtype=np.int32), dtype=wp.int32, device='cpu'))
    tester = SimpleNamespace(origin=origin, mesh=mesh, device='cpu')
    points = origin + np.array([[.2, .2, .0021], [.2, .2, -.01], [1., 1., 0.]])
    distance = mesh_distance(tester, points)
    np.testing.assert_allclose(distance, [.0001, .01, np.sqrt(.5)], atol=1e-6)


def test_float64_narrow_band_preserves_skinny_faces_and_open_boundaries():
    import warp as wp
    from mesh_to_gs.visibility import mesh_near_surface
    wp.config.quiet = True
    wp.init()
    origin = np.array([100., 130., 1.])
    # Width is lost in a float32 upload at these local coordinates. The BVH
    # still has a conservative box; float64 vertices must supply the distance.
    local = np.array([[8., 8., 0.], [8.1, 8., 0.], [8., 8. + 1e-8, 0.]])
    mesh = wp.Mesh(points=wp.array(local.astype(np.float32), dtype=wp.vec3, device='cpu'),
                   indices=wp.array(np.arange(3, dtype=np.int32), dtype=wp.int32, device='cpu'))
    tester = SimpleNamespace(origin=origin, mesh=mesh, device='cpu')
    tri = (local + origin)[None]
    points = origin + np.array([[8.02, 8. + 1e-9, 0.], [8.02, 8., .0003], [8.2, 8., 0.]])
    np.testing.assert_array_equal(mesh_near_surface(tester, tri, points, .0002), [True, False, False])
