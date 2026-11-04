"""Boundary constraints rebuilt for support optimization agree with the initializer."""
import numpy as np
from scipy.spatial.transform import Rotation

from mesh_to_gs.surface import charts, sample_component, covariance
from mesh_to_gs.support_constraints import chart_boundaries, constraints, project


def square(subdivisions=8):
    xy = np.mgrid[0:subdivisions + 1, 0:subdivisions + 1].reshape(2, -1).T / subdivisions
    vertices = np.column_stack((xy, np.zeros(len(xy))))
    ids = []
    for i in range(subdivisions):
        for j in range(subdivisions):
            a = i * (subdivisions + 1) + j
            b = a + subdivisions + 1
            ids.extend([[a, b, a + 1], [b, b + 1, a + 1]])
    ids = np.array(ids)
    return charts(vertices[ids], ids)


def frame(q):
    r = Rotation.from_quat(q, scalar_first=True).as_matrix()
    return r[:, :, 0], r[:, :, 1], r[:, :, 2]


def test_boundaries_are_the_outer_rim_only():
    s = square()
    a, b, owner, _ = chart_boundaries(s.triangles, s.labels)
    assert len(a) == 32 and np.all(owner == 0)
    on_rim = lambda p: np.isclose(p[:, 0], 0) | np.isclose(p[:, 0], 1) | np.isclose(p[:, 1], 0) | np.isclose(p[:, 1], 1)
    assert on_rim(a).all() and on_rim(b).all()


def test_initial_supports_satisfy_rebuilt_constraints_and_projection_restores_them():
    s = square()
    p, n, _, edges, h, _ = sample_component(s, 0, 2048, 3)
    sc, q, _ = covariance(p, n, edges, h)
    t, bt, normal = frame(q)
    sigma = sc[:, :2].astype(np.float64)
    m, limit = constraints(p, normal, t, bt, np.zeros(len(p), np.int64),
                           chart_boundaries(s.triangles, s.labels), radius=3 * sigma.max(1) + 1e-3)
    C = np.zeros((len(p), 2, 2))
    C[:, 0, 0], C[:, 1, 1] = sigma[:, 0] ** 2, sigma[:, 1] ** 2
    variance = np.einsum('nki,nij,nkj->nk', m, C, m)
    used = np.isfinite(limit)
    assert used.any(axis=1).mean() > .2
    assert np.all(variance[used] <= (limit[used] ** 2) * (1 + 1e-3))
    # Inflate every support 1.5x: projection must bring it back inside.
    D = project(C * 2.25, m.astype(np.float64), limit.astype(np.float64))
    after = np.einsum('nki,nij,nkj->nk', m, D, m)
    assert np.all(after[used] <= (limit[used] ** 2) * (1 + 1e-6))
    # Check against the true geometry: 2 sigma along x and y stays within the square.
    R = np.stack((t, bt), axis=2)
    W = np.einsum('nik,nkl,njl->nij', R, D, R)
    for axis in (0, 1):
        distance = np.minimum(p[:, axis], 1 - p[:, axis])
        assert np.max(2 * np.sqrt(W[:, axis, axis]) - distance) < 1e-6


def test_seam_multiple_zero_drops_seam_constraints():
    s = square(2)
    a, b, owner, _ = chart_boundaries(s.triangles, s.labels)
    seam = np.isclose(a[:, 0], 1) & np.isclose(b[:, 0], 1)          # the x = 1 side
    p = np.array([[.95, .5, 0.]])
    n = np.array([[0, 0, 1.]])
    t, bt = np.array([[1, 0, 0.]]), np.array([[0, 1, 0.]])
    m, limit = constraints(p, n, t, bt, np.zeros(1, np.int64), (a, b, owner, seam), radius=np.array([.2]))
    assert np.isclose(np.nanmin(np.where(np.isfinite(limit), limit, np.nan)), .025)
    m, limit = constraints(p, n, t, bt, np.zeros(1, np.int64), (a, b, owner, seam), radius=np.array([.2]), seam_multiple=0)
    assert not np.isfinite(limit).any()


def test_one_projection_sweep_satisfies_every_constraint():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5000, 2, 2))
    C = A @ A.transpose(0, 2, 1) + .01 * np.eye(2)
    angle = rng.uniform(0, np.pi, (5000, 8))
    m = np.stack((np.cos(angle), np.sin(angle)), -1)
    limit = rng.uniform(.1, 2, (5000, 8))
    D = project(C, m, limit)
    variance = np.einsum('nki,nij,nkj->nk', m, D, m)
    assert np.all(variance <= limit ** 2 * (1 + 1e-9))
    assert np.all(np.linalg.eigvalsh(D) > -1e-12)          # still positive semidefinite


