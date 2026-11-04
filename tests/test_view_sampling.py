"""Priority view sampling and the retained Mesh minibatch RNG convention."""

import numpy as np
import pytest

from mesh_to_gs.view_sampling import ViewPriorities, detail_score, training_order


def test_detail_measure_ignores_silhouettes_but_detects_interior_texture():
    blank = np.full((20, 20, 3), .5)
    texture = np.repeat((np.indices((20, 20)).sum(0) % 2)[..., None], 3, axis=2)
    assert detail_score(texture) > detail_score(blank)
    foreground = np.zeros((20, 20), bool)
    foreground[4:16, 4:16] = True
    cutout = np.ones_like(blank)
    cutout[foreground] = .3
    assert detail_score(cutout, foreground) < 1e-12
    assert detail_score(cutout, np.zeros_like(foreground)) == 0


def test_view_importance_keeps_uniform_floor_and_rewards_informative_views():
    sampler = ViewPriorities([0., 1., 10.], [0., 1., 10.], .5)
    p = sampler.probabilities()
    assert np.isclose(p.sum(), 1) and (p >= .5/3).all()
    assert p[2] > p[1] > p[0]


def test_loss_priority_follows_learning_and_records_actual_updates():
    sampler = ViewPriorities([1., 1.], [0., 0.], .5, ema=0.)
    sampler.update(0, 0.)
    assert sampler.probabilities()[0] < sampler.probabilities()[1]
    assert sampler.counts.tolist() == [1, 0]
    assert sampler.initial_losses.tolist() == [1., 1.]


def test_zero_information_is_uniform_and_bad_scores_are_rejected():
    sampler = ViewPriorities([0., 0.], [0., 0.])
    np.testing.assert_allclose(sampler.probabilities(), [.5, .5])
    with pytest.raises(ValueError):
        ViewPriorities([np.nan], [0.])


def test_uniform_order_matches_legacy_and_duration_runs_share_minibatch_prefix():
    legacy = np.random.default_rng(20261102)
    expected = np.concatenate([legacy.permutation(24) for _ in range(500//24+1)])[:500]
    order, rng = training_order(20261102, 24, 500)
    np.testing.assert_array_equal(order, expected)
    np.testing.assert_array_equal(rng.integers(1000, size=100), legacy.integers(1000, size=100))
    short, short_rng = training_order(42, 24, 500, 500)
    long, long_rng = training_order(42, 24, 3000, 500)
    np.testing.assert_array_equal(short, long[:500])
    np.testing.assert_array_equal(short_rng.integers(1000, size=100), long_rng.integers(1000, size=100))






