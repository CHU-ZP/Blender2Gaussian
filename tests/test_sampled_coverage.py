"""Finite-probe sampling must preserve ownership and report actual failures."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mesh_to_gs.sampled_coverage import SampledConfig, SampledSurface, distributed_rows


def patch(subdivisions=1, size=.1):
    v = np.array([[x, y, 0.] for y in np.linspace(0, size, subdivisions + 1)
                  for x in np.linspace(0, size, subdivisions + 1)])
    ids = []
    for y in range(subdivisions):
        for x in range(subdivisions):
            a = y * (subdivisions + 1) + x
            ids.extend([[a, a + 1, a + subdivisions + 1],
                        [a + 1, a + subdivisions + 2, a + subdivisions + 1]])
    ids = np.array(ids)
    return v[ids], ids


@pytest.mark.parametrize('options', [dict(texture_weight=np.nan), dict(texture_weight=np.inf),
    dict(batch_size=1.5), dict(repair_rounds=1.5), dict(seed=-1), dict(neighbours=True),
    dict(spacing=0), dict(threshold=np.nan), dict(failure_fraction=np.nan)])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        SampledConfig(**options)


def test_tiny_chart_allocation_preserves_owner_and_local_distribution():
    weights = np.array([1e20, .25, .75, 2.])
    owner = np.array([0, 1, 1, 2])
    counts = np.array([1, 100, 4])
    rows, labels, fraction = distributed_rows(weights, owner, counts, np.random.default_rng(1), return_fraction=True)
    np.testing.assert_array_equal(owner[rows], labels)
    assert (rows == 1).sum() == 25 and (rows == 2).sum() == 75
    assert np.all((fraction >= 0) & (fraction <= 1))


def test_retriangulation_does_not_allocate_per_face_and_keeps_thin_charts():
    count = []
    for n in (1, 20):
        s = SampledSurface(*patch(n))
        face, bary, weights, boundary = s.probes(.5, 1)
        count.append((~boundary).sum())
        np.testing.assert_allclose(weights[~boundary].sum(), .01)
        np.testing.assert_allclose(bary.sum(1), 1.)
    assert count[0] == count[1] < 800
    t, ids = patch()
    thin = t * [1., 1e-14, 1.] + [0., 0., 1e-5]
    s = SampledSurface(np.r_[t, thin], np.r_[ids, ids + 4])
    face, _, _, boundary = s.probes(1., 10)
    assert set(s.surface.labels[face[~boundary]]) == {0, 1}
    assert len(s.surface.areas) == 4


def test_exact_corner_keeps_both_boundary_constraints():
    s = SampledSurface(*patch())
    s.append(np.array([0]), np.array([[1., 0, 0]]), np.array([True]))
    C = (s.rotations[0] * s.scales[0] ** 2) @ s.rotations[0].T
    assert np.sqrt(C[0, 0]) <= s.config.edge_tolerance / s.config.support_radius * (1 + 1e-10)
    assert np.sqrt(C[1, 1]) <= s.config.edge_tolerance / s.config.support_radius * (1 + 1e-10)
    np.testing.assert_allclose(s.evaluate(np.array([0]), np.array([[1., 0, 0]]))[0], .99)


def test_full_radius_fallback_finds_distant_support_and_excludes_other_sheets():
    t, ids = patch()
    s = SampledSurface(np.r_[t, t + [0, 0, .00001]], np.r_[ids, ids + 4],
                       config=SampledConfig(neighbours=1, normal_fraction=100.))
    s.append(np.array([0, 0]), np.array([[.2, .4, .4], [0., .5, .5]]), np.zeros(2, bool))
    s.scales[:] = [[1e-5] * 3, [.2] * 3]
    f, b = np.array([0, 2]), np.array([[.18, .41, .41]] * 2)
    value, queries = s.evaluate(f, b)
    assert value[0] > .95 and value[1] == 0
    assert queries['full_radius_fallbacks'] == 2


def test_endpoint_failure_cannot_pass_with_zero_measure(monkeypatch):
    s = SampledSurface(*patch())
    monkeypatch.setattr(s, 'probes', lambda *a: (np.array([0, 0]), np.array([[1/3]*3, [1., 0, 0]]),
                                               np.array([1., 0.]), np.array([False, True])))
    monkeypatch.setattr(s, 'evaluate', lambda *a, **kw: (np.array([.99, 0.]), {}))
    report, failures = s.validate()
    assert report['status'] == 'failed' and report['endpoints']['failed_probes'] == 1
    assert len(failures['face']) == 1


def test_repair_closes_fit_probes_without_using_validation_samples(tmp_path):
    s = SampledSurface(*patch()).fit()
    assert s.rounds[0]['failed_probes'] > 0 and s.rounds[-1]['failed_probes'] == 0
    before = s.points.copy()
    report = s.export(tmp_path)
    np.testing.assert_array_equal(s.points, before)
    assert report['validation_stage'] == 'serialized float64 PLY'
    assert report['continuous_certificate'] is False
    assert report['status'] == 'failed'  # fit success does not certify new probes
    assert np.load(tmp_path/'failed_probes.npz')['face'].size > 0
    from mesh_to_gs.coverage import read_model
    p, sc, r, a = read_model(tmp_path/'model.ply')
    np.testing.assert_allclose(p, s.points, atol=0, rtol=0)
    np.testing.assert_allclose(sc, s.scales, rtol=1e-14)
    # Validation must use serialized opacity, not the initializer's config.
    degraded, _ = s.validate(model=(p, sc, r, np.zeros_like(a)))
    assert degraded['failed_probes'] == degraded['probes']


def test_finite_validation_can_pass_and_respects_round_limit():
    cfg = SampledConfig(edge_tolerance=.05, overlap=2.)
    s = SampledSurface(*patch(), config=cfg).fit()
    assert s.validate()[0]['status'] == 'passed'
    limited = SampledSurface(*patch(), config=replace(cfg, overlap=.1, repair_rounds=0)).fit()
    assert len(limited.rounds) == 1 and limited.rounds[0]['failed_probes'] > 0
    assert limited.validate()[0]['status'] == 'failed'


def test_empty_surface_and_texture_without_uv_are_explicit_errors():
    with pytest.raises(ValueError, match='nondegenerate'):
        SampledSurface(np.zeros((1, 3, 3)), np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match='UV'):
        SampledSurface(*patch(), texture=SimpleNamespace(image=np.ones((2, 2, 3))))


def test_constant_texture_skips_gradients_but_exports_uv_colour(tmp_path):
    t, ids = patch()
    texture = SimpleNamespace(image=np.full((2, 2, 3), .25))
    s = SampledSurface(t, ids, uv=t[:, :, :2], texture=texture,
                       config=SampledConfig(edge_tolerance=.05, overlap=2.)).fit()
    assert s.jacobian is None and s.constant_texture
    np.testing.assert_allclose(s.colours, .25)
    assert s.export(tmp_path)['texture_mode'] == 'global_range'
