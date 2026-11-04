import numpy as np

from mesh_to_gs.coverage import CoverageConfig, Sampler, certify_model, probe_model, read_model
from mesh_to_gs.coverage_repair import closest_chart_points, seed_sampler
from test_coverage import square


def test_projection_stays_in_owning_chart_and_clamps_boundaries():
    triangles, faces = square(.01, z=0)
    triangles = np.concatenate((triangles, triangles + [0, 0, .001]))
    faces = np.concatenate((faces, faces + 4))
    sampler = Sampler(triangles, faces)
    p = np.array([[.003, .002, .00099], [.012, .003, .00099], [.008, .009, .00099]])
    q, face, bary = closest_chart_points(p, np.zeros(3, np.int64), sampler.surface)
    np.testing.assert_allclose(q, [[.003, .002, 0], [.01, .003, 0], [.008, .009, 0]], atol=1e-15)
    assert (sampler.surface.labels[face] == 0).all()
    assert face[2] == 1  # Cross the internal diagonal within the same chart.
    assert (bary >= 0).all()
    np.testing.assert_allclose(bary.sum(1), 1)


def test_low_opacity_optimized_seeds_repair_without_losing_learned_colour(tmp_path):
    triangles, faces = square(.008, z=0)
    cfg = CoverageConfig(edge_tolerance=.004, max_spacing=.02)
    sampler = Sampler(triangles, faces, config=cfg)
    p = np.array([[.004, .004, .002]])
    scales = np.array([[.001, .001, .00002]])
    rotations = np.eye(3)[None]
    opacity = np.array([.4])
    colour = sampler.colour[None] + .004
    stats = seed_sampler(sampler, p, scales, rotations, opacity, colour, np.array([0]))
    assert stats['maximum_projection_distance'] == .002
    sampler.export(tmp_path / 'raw')
    assert probe_model(tmp_path / 'raw/model.ply', tmp_path / 'raw/surface', cfg)['failed_probes'] > 0
    sampler.run()
    report = sampler.export(tmp_path / 'repaired')
    assert report['status'] == 'complete'
    assert len(sampler.points) > 1
    np.testing.assert_allclose(sampler.colours[0], colour[0])
    assert sampler.opacities[0] == .4
    assert read_model(tmp_path / 'repaired/model.ply')[3][0] == .4
    assert certify_model(tmp_path / 'repaired/model.ply', tmp_path / 'repaired/surface', cfg)['status'] == 'complete'
    assert probe_model(tmp_path / 'repaired/model.ply', tmp_path / 'repaired/surface', cfg)['failed_probes'] == 0


def test_seed_checkpoint_resumes_before_deficit_repair(tmp_path):
    triangles, faces = square(.008, z=0)
    cfg = CoverageConfig(edge_tolerance=.004, checkpoint_cells=1)
    params = (np.array([[.002, .002, .001], [.006, .006, .001]]),
              np.full((2, 3), .001), np.tile(np.eye(3), (2, 1, 1)),
              np.array([.6, .7]), np.full((2, 3), .21), np.array([0, 1]))
    sampler = Sampler(triangles, faces, config=cfg)
    state = tmp_path / 'pending.npz'
    def interrupt(_):
        raise KeyboardInterrupt
    import pytest
    with pytest.raises(KeyboardInterrupt):
        seed_sampler(sampler, *params, checkpoint=state, progress=interrupt)
    resumed = Sampler(triangles, faces, config=cfg, state=state)
    seed_sampler(resumed, *params, checkpoint=state)
    fresh = Sampler(triangles, faces, config=cfg)
    seed_sampler(fresh, *params)
    resumed.run(); fresh.run()
    np.testing.assert_array_equal(resumed.points, fresh.points)
    np.testing.assert_array_equal(resumed.colours, fresh.colours)


def test_repair_limits_near_one_opacity_but_retains_low_learned_values():
    triangles, faces = square(.008, z=0)
    sampler = Sampler(triangles, faces)
    sampler.add(0, np.array([1/3] * 3), 'optimized', opacity=.9999)
    sampler.add(0, np.array([1/3] * 3), 'optimized', opacity=.3)
    assert sampler.opacities[0] == sampler.config.opacity
    assert sampler.opacities[1] == .3


def test_repaired_opacity_prevents_renderer_early_termination_hole():
    import pytest
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from scripts.render_gaussians import configure_cuda
    configure_cuda()
    from gsplat import rasterization
    triangles, faces = square(.008, z=0)
    sampler = Sampler(triangles, faces)
    sampler.add(0, np.array([1/3] * 3), 'optimized', opacity=.91)
    sampler.add(0, np.array([1/3] * 3), 'optimized', opacity=.9999)
    cuda = lambda x: torch.tensor(x, dtype=torch.float32, device='cuda')
    _, alpha, _ = rasterization(
        means=cuda([[0, 0, 2], [0, 0, 2.1]]), quats=cuda([[1, 0, 0, 0]] * 2),
        scales=cuda([[.01] * 3] * 2), opacities=cuda(np.asarray(sampler.opacities)),
        colors=cuda([[1] * 3] * 2), viewmats=cuda(np.eye(4)[None]),
        Ks=cuda([[[560, 0, 32.5], [0, 560, 32.5], [0, 0, 1]]]),
        width=64, height=64, packed=True, eps2d=.05, rasterize_mode='classic')
    assert float(alpha[0, 32, 32, 0]) > .99
