"""Continuous coverage, conservative candidates and uncapped repair contracts."""
from dataclasses import asdict

import numpy as np
import pytest

from mesh_to_gs.coverage import (CoverageConfig, Sampler, cell_lower_bounds, certify_model, energy, footprint_box,
                               outside_upper_bound, spatial_index, support, texture_bounds,
                               triangle_lower_bound)


def square(size=.04, z=3., subdivisions=1):
    vertices = np.array([[x, y, z] for y in np.linspace(0, size, subdivisions + 1)
                         for x in np.linspace(0, size, subdivisions + 1)])
    faces = []
    for y in range(subdivisions):
        for x in range(subdivisions):
            a = y * (subdivisions + 1) + x
            b, c, d = a + 1, a + subdivisions + 1, a + subdivisions + 2
            faces.extend([[a, b, c], [b, d, c]])
    return vertices[np.array(faces)], np.array(faces)


def test_triangle_bound_covers_continuous_interior_not_separate_vertices():
    rng = np.random.default_rng(251)
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]])
    p = tri.copy()
    sc = np.full((3, 3), .03)
    r = np.tile(np.eye(3), (3, 1, 1))
    a = np.full(3, .99)
    assert support(energy(tri, p, sc, r), a).min() > .98
    assert triangle_lower_bound(tri, p, sc, r, a) == 0
    for _ in range(20):
        p = rng.uniform(-.2, 1.2, (8, 3)); p[:, 2] = 0
        sc = rng.uniform(.2, 2, (8, 3))
        r = np.tile(np.eye(3), (8, 1, 1))
        a = rng.uniform(.5, .99, 8)
        bary = rng.dirichlet([1, 1, 1], 1000)
        assert support(energy(bary @ tri, p, sc, r), a).min() >= triangle_lower_bound(tri, p, sc, r, a)
        assert support(energy(bary @ tri, p, sc, r), a).max() <= outside_upper_bound(tri, p, sc, r, a)


def test_aabb_query_keeps_large_footprints_with_distant_centres():
    index = spatial_index()
    index.insert(0, footprint_box(np.array([10., 0, 0]), np.array([5., .1, .1]), np.eye(3), 3))
    assert list(index.intersection((0, 0, 0, .01, .01, .01))) == [0]


def test_batched_bounds_keep_ragged_candidates_and_conservative_interior():
    rng = np.random.default_rng(20261104)
    vertices = rng.uniform(-.1, .1, (7, 3, 3))
    centers = rng.uniform(-.2, .2, (200, 3))
    scales = rng.uniform(.02, .3, (200, 3))
    rotations = np.tile(np.eye(3), (200, 1, 1))
    opacity = rng.uniform(.01, .99, 200)
    ids = [np.empty(0, np.int64), np.arange(200), np.arange(10), np.empty(0, np.int64),
           np.array([199]), np.arange(30, 120), np.empty(0, np.int64)]
    bounds = cell_lower_bounds(vertices, ids, centers, scales, rotations, opacity)
    for face, candidates in enumerate(ids):
        scalar = triangle_lower_bound(vertices[face], centers[candidates], scales[candidates],
                                      rotations[candidates], opacity[candidates])
        assert abs(bounds[face] - scalar) < 1e-14
        interior = rng.dirichlet([1, 1, 1], 1000) @ vertices[face]
        actual = support(energy(interior, centers[candidates], scales[candidates], rotations[candidates]), opacity[candidates])
        assert actual.min() >= bounds[face]


def test_bilinear_bounds_converge_inside_texel_and_across_wrap():
    from mesh_to_gs.adaptive import bilinear
    image = np.array([[[0., 0., 0.], [1., .5, .2]], [[.3, .4, .5], [.7, .2, .1]]])
    rng = np.random.default_rng(3)
    for low, high in (([.2499, .2499], [.2501, .2501]), ([-.2, .9], [.2, 1.2])):
        lo, hi = texture_bounds(image, np.array(low), np.array(high))
        value = bilinear(image, rng.uniform(low, high, (1000, 2)))
        assert np.all(value >= lo - 1e-14) and np.all(value <= hi + 1e-14)
    lo, hi = texture_bounds(image, np.array([.2499, .2499]), np.array([.2501, .2501]))
    assert (hi - lo).max() < .001


def test_uncapped_sampling_export_edges_and_stale_model(tmp_path):
    assert not any('budget' in k or 'count' in k or 'growth' in k for k in asdict(CoverageConfig()))
    t, f = square()
    sampler = Sampler(t, f, config=CoverageConfig(max_spacing=.02, edge_tolerance=.004)).run()
    report = sampler.export(tmp_path)
    assert report['status'] == 'complete'
    assert report['minimum_lower_bound'] >= .95
    assert report['partition_complete']
    # Independent dense probes include every edge and corner.
    vertices = np.array([[x, y, 3] for x in np.linspace(0, .04, 81) for y in np.linspace(0, .04, 81)])
    value = support(energy(vertices, np.array(sampler.points), np.array(sampler.scales),
                           np.array(sampler.rotations)), np.array(sampler.opacities))
    assert value.min() >= .95
    outside = np.array([[-.00401, .02, 3], [.02, -.00401, 3], [-.003, -.003, 3],
                        [.043, .043, 3]])
    value = support(energy(outside, np.array(sampler.points), np.array(sampler.scales),
                           np.array(sampler.rotations)), np.array(sampler.opacities))
    assert value.max() == 0
    from plyfile import PlyData
    data = PlyData.read(tmp_path / 'model.ply', mmap=False)
    data['vertex'].data['opacity'][:] = -20
    data.write(tmp_path / 'model.ply')
    assert certify_model(tmp_path / 'model.ply', tmp_path / 'surface', sampler.config)['status'] == 'incomplete'


def test_checkpoint_resume_exact_and_targets_cannot_disappear(tmp_path):
    t, f = square(.015)
    cfg = CoverageConfig(max_spacing=.02, edge_tolerance=.003, checkpoint_cells=25)
    a = Sampler(t, f, config=cfg)
    checkpoint = tmp_path / 'state.npz'
    def stop(_):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        a.run(checkpoint=checkpoint, progress=stop)
    b = Sampler(t, f, config=cfg, state=checkpoint).run()
    c = Sampler(t, f, config=cfg).run()
    np.testing.assert_array_equal(b.points, c.points)
    report = b.export(tmp_path / 'model')
    assert report['status'] == 'complete'
    target = tmp_path / 'model' / 'surface'
    cf = np.load(target / 'certificate_face.npy')
    cb = np.load(target / 'certificate_barycentric.npy')
    np.save(target / 'certificate_face.npy', cf[1:])
    np.save(target / 'certificate_barycentric.npy', cb[1:])
    assert not certify_model(tmp_path / 'model' / 'model.ply', target, cfg)['partition_complete']
    with pytest.raises(ValueError, match='changed'):
        Sampler(t + [0, 0, .1], f, config=cfg, state=checkpoint)


def test_unrelated_parallel_sheet_cannot_certify_and_sliver_survives():
    t, f = square(.01)
    t = np.concatenate((t, t + [0, 0, .000001]))
    f = np.concatenate((f, f + 4))
    sampler = Sampler(t, f, config=CoverageConfig(edge_tolerance=.004))
    sampler.add(0, np.array([1/3] * 3))
    assert len(sampler.candidates(t[2], sampler.surface.labels[2])) == 0
    sliver = np.array([[[0., 0., 3], [.1, 0, 3], [.05, 1e-14, 3]]])
    s = Sampler(sliver, np.array([[0, 1, 2]]))
    assert len(s.surface.areas) == 1 and s.surface.skipped == 0


def test_retriangulation_does_not_require_one_sample_per_face():
    t, f = square(.01, subdivisions=12)
    sampler = Sampler(t, f, config=CoverageConfig(edge_tolerance=.004)).run()
    assert len(sampler.points) < len(t) / 2
    assert set(sampler.cert_faces) == set(range(len(t)))


def test_colour_certificate_reads_serialized_rgb_and_directional_texture(tmp_path):
    t, f = square(.01)
    uv = t[:, :, :2] / .01
    image = np.array([[[.2, .3, .4], [.24, .3, .4]], [[.2, .3, .4], [.24, .3, .4]]])
    sampler = Sampler(t, f, uv, image, config=CoverageConfig(edge_tolerance=.004)).run()
    report = sampler.export(tmp_path)
    assert report['status'] == 'complete'
    assert report['quality']['maximum_footprint_colour_error_linear'] <= .02
    from plyfile import PlyData
    data = PlyData.read(tmp_path / 'model.ply', mmap=False)
    data['vertex'].data['f_dc_0'][:] = 1.5
    data.write(tmp_path / 'model.ply')
    assert sampler.validate_quality(tmp_path / 'model.ply')['status'] == 'incomplete'


def test_texture_noise_within_tolerance_does_not_force_microscopic_scales():
    t, f = square(.04)
    cfg = CoverageConfig(colour_error=.08)
    # Many repeats give a large local derivative, while every source colour
    # remains within tolerance on the entire unconstrained footprint.
    uv = t[:, :, :2] / .04 * 100
    image = np.array([[[.3, .3, .3], [.33, .3, .3]],
                      [[.3, .3, .3], [.33, .3, .3]]])
    textured = Sampler(t, f, uv, image, config=cfg)
    constant = Sampler(t, f, config=cfg)
    bary = np.array([.2, .3, .5])
    textured.add(0, bary)
    constant.add(0, bary)
    np.testing.assert_allclose(textured.scales, constant.scales, rtol=1e-12)
    covariance = (textured.rotations[0] * textured.scales[0]) @ (textured.rotations[0] * textured.scales[0]).T
    assert textured.footprint_colour_error(textured.points[0], covariance, 0, textured.colours[0]) < cfg.colour_error


def test_low_contrast_repeat_texture_has_a_global_certificate(tmp_path, monkeypatch):
    from mesh_to_gs.adaptive import bilinear
    import mesh_to_gs.coverage as coverage
    t, f = square(.01)
    uv = t[:, :, :2] / .01 * 1000
    rng = np.random.default_rng(812)
    image = rng.uniform(.3, .31, (16, 16, 3))
    cfg = CoverageConfig(edge_tolerance=.004)
    textured = Sampler(t, f, uv, image, config=cfg)
    def unexpected_scan(*_):
        raise AssertionError('A valid global certificate must avoid per-footprint texel scans')
    monkeypatch.setattr(coverage, 'texture_bounds', unexpected_scan)
    textured.run()
    constant = Sampler(t, f, config=cfg).run()
    np.testing.assert_array_equal(textured.points, constant.points)
    np.testing.assert_array_equal(textured.scales, constant.scales)
    # The cached bound covers arbitrary repeats, not just the birth face UVs.
    actual = bilinear(image, rng.uniform(-100, 100, (2000, 2)))
    for point, rotation, scale, face, colour in zip(textured.points, textured.rotations,
                                                  textured.scales, textured.faces, textured.colours):
        covariance = (rotation * scale) @ (rotation * scale).T
        bound = textured.footprint_colour_error(point, covariance, face, colour)
        assert np.abs(actual - colour).max() <= bound <= cfg.colour_error
    report = textured.export(tmp_path)
    assert report['status'] == report['quality']['status'] == 'complete'


def test_high_contrast_texture_still_uses_tight_local_bounds():
    from mesh_to_gs.adaptive import bilinear
    t, f = square(.01, z=0.)
    uv = t[:, :, :2] / .01
    image = np.array([[[.1, .1, .1], [.9, .9, .9]],
                      [[.1, .1, .1], [.9, .9, .9]]])
    sampler = Sampler(t, f, uv, image)
    point = np.array([.0025, .0025, 0.])
    colour = bilinear(image, np.array([[.25, .25]]))[0]
    covariance = np.diag([1e-12, 1e-12, 4e-16])
    bound = sampler.footprint_colour_error(point, covariance, 0, colour)
    # A distant colour hotspot cannot force rejection of a tiny local footprint.
    assert bound < .001
    actual_uv = np.array([[.2499, .25], [.2501, .25]])
    assert np.abs(bilinear(image, actual_uv) - colour).max() <= bound


def test_coverage_cli_rejects_budget_and_local_render_preserves_precision(tmp_path):
    from scripts.initialize_coverage_surface import parse_args
    from scripts.render_gaussians import centered_view_matrix
    for option in ('--budget', '--max-budget', '--max-growth'):
        with pytest.raises(SystemExit):
            parse_args(['scene.blend', '--output-dir', str(tmp_path / 'out'), option, '10'])
    origin = np.array([120., -80., 30.])
    matrix = np.eye(4)
    matrix[:3, 3] = -origin + [0, 0, 3]
    delta = np.array([[1e-6, -1e-6, 0.]])
    local = (origin + delta - origin).astype('f4')
    camera = centered_view_matrix(matrix, origin).astype('f4')
    np.testing.assert_allclose(local @ camera[:3, :3].T + camera[:3, 3], delta + [0, 0, 3], atol=1e-12)


def test_streamed_groups_recheck_whole_combined_target(tmp_path):
    from scripts.initialize_coverage_surface import combine, parse_args
    cfg = CoverageConfig(edge_tolerance=.004)
    completed = []
    for i, z in enumerate((3., 3.1)):
        t, f = square(.005, z)
        sampler = Sampler(t, f, config=cfg).run()
        directory = tmp_path / 'groups' / str(i)
        report = sampler.export(directory)
        completed.append(dict(directory=str(directory.relative_to(tmp_path)), coverage=report,
                              object=f'object{i}', material='colour', colour=[.5] * 3, has_uv=False, image=None))
    args = parse_args(['scene.blend', '--output-dir', str(tmp_path), '--edge-tolerance', '.004'])
    report = combine(tmp_path, completed, 'source', args, [[0, 0, 3], [.005, .005, 3.1]])
    independent = certify_model(tmp_path / 'model.ply', tmp_path / 'surface', cfg)
    assert independent['status'] == 'complete'
    assert report['target_sha256'] == independent['target_sha256']
    assert report['model_sha256'] == independent['model_sha256']
    assert independent['target_faces'] == 4


def test_shared_uv_edge_with_different_jacobians_uses_neighbour_mapping():
    from mesh_to_gs.adaptive import bilinear
    t, f = square(.01, z=0.)
    uv = t[:, :, :2] / .01
    uv[1, 1] = [3, 3]
    image = np.tile(np.linspace(.1, .8, 8)[None, :, None], (8, 1, 3))
    sampler = Sampler(t, f, uv, image)
    assert len(sampler.surface.component_areas) == 1
    center = np.array([.0049, .0049, 0.])
    source_colour = bilinear(image, np.array([[.49, .49]]))[0]
    neighbor = np.array([.0052, .0052, 0.])
    actual_uv = uv[1, 0] + sampler.jacobians[1] @ (neighbor - t[1, 0])
    actual_error = np.abs(bilinear(image, actual_uv[None])[0] - source_colour).max()
    covariance = np.diag([.0002 ** 2, .0002 ** 2, .000004 ** 2])
    assert sampler.footprint_colour_error(center, covariance, 0, source_colour) >= actual_error
    # The original face's affine continuation would underestimate this error.
    lo, hi = texture_bounds(image, np.array([.43, .43]), np.array([.55, .55]))
    assert np.maximum(np.abs(lo - source_colour), np.abs(hi - source_colour)).max() < actual_error
    # The two source triangles share a full AABB. An ellipse well inside the
    # first triangle must not inherit the other's extrapolated UV colours.
    center = np.array([.001, .001, 0.])
    colour = bilinear(image, np.array([[.1, .1]]))[0]
    covariance = np.diag([1e-12, 1e-12, 4e-16])
    assert sampler.footprint_colour_error(center, covariance, 0, colour) < .001


def test_allocation_failure_keeps_correspondences_resumable(tmp_path, monkeypatch):
    t, f = square(.005)
    sampler = Sampler(t, f, config=CoverageConfig(edge_tolerance=.004))
    def fail(_):
        raise MemoryError('simulated resource failure')
    monkeypatch.setattr(sampler.kinds, 'reserve', fail)
    state = tmp_path / 'pending.npz'
    with pytest.raises(MemoryError):
        sampler.run(checkpoint=state)
    resumed = Sampler(t, f, config=sampler.config, state=state).run()
    assert resumed.export(tmp_path / 'model')['status'] == 'complete'


@pytest.mark.parametrize('column', ['scales', 'cert_bounds'])
def test_interrupt_during_column_write_rolls_back_and_resumes(tmp_path, monkeypatch, column):
    t, f = square(.005)
    cfg = CoverageConfig(edge_tolerance=.004)
    sampler = Sampler(t, f, config=cfg)
    append = getattr(sampler, column).append
    def interrupt(value):
        append(value)
        raise KeyboardInterrupt
    monkeypatch.setattr(getattr(sampler, column), 'append', interrupt)
    state = tmp_path / 'pending.npz'
    with pytest.raises(KeyboardInterrupt):
        sampler.run(checkpoint=state)
    assert len(sampler.points) == len(sampler.scales) == len(sampler.faces)
    assert len(sampler.cert_faces) == len(sampler.cert_bary) == len(sampler.cert_bounds)
    if column == 'scales':
        assert len(sampler.points) == len(sampler.candidates(t[0])) == 0
    resumed = Sampler(t, f, config=cfg, state=state).run()
    fresh = Sampler(t, f, config=cfg).run()
    np.testing.assert_array_equal(resumed.points, fresh.points)
    np.testing.assert_array_equal(resumed.cert_bary, fresh.cert_bary)
    assert resumed.export(tmp_path / 'model')['status'] == 'complete'


def test_tiny_positive_face_uv_mapping_is_not_softened_by_area_floor():
    triangle = np.array([[[0, 0, 0.], [1e-16, 0, 0], [0, 1e-16, 0]]])
    uv = np.array([[[0, 0.], [1, 0], [0, 1]]])
    sampler = Sampler(triangle, np.array([[0, 1, 2]]), uv)
    reconstructed = (triangle[0] - triangle[0, 0]) @ sampler.jacobians[0].T + uv[0, 0]
    np.testing.assert_allclose(reconstructed, uv[0], atol=1e-14)
