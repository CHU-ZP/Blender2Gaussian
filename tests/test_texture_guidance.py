import numpy as np
from scipy.spatial.transform import Rotation

from mesh_to_gs.adaptive import TextureFeatures
from mesh_to_gs.support_constraints import project
from mesh_to_gs.texture_guidance import structure_tensor, guided_covariance, covariance_parameters


def stripe_tensor():
    x = (np.arange(128)+.5)/128
    image = np.tile((127+100*np.sin(12*np.pi*x)).astype('uint8')[None, :, None], (128, 1, 3))
    texture = TextureFeatures(image)
    uv = np.array([[.5, .5]])
    jac = np.array([[[1., 0, 0], [0, 1, 0]]])
    t, b = np.array([[1., 0, 0]]), np.array([[0., 1, 0]])
    return texture, uv, jac, t, b


def test_stripe_long_axis_and_area_preservation():
    f, uv, j, t, b = stripe_tensor()
    tensor = structure_tensor(f, uv, j, t, b, .05)
    C, gate = guided_covariance(np.array([[.04, .04]]), tensor, .05, 1)
    assert gate[0] > .5 and C[0, 1, 1] > C[0, 0, 0]
    np.testing.assert_allclose(np.linalg.det(C), .04**4, rtol=1e-12)


def test_physical_scaling_and_rotation_equivariance():
    f, uv, j, t, b = stripe_tensor()
    a = structure_tensor(f, uv, j, t, b, .05)
    np.testing.assert_allclose(a, 100*structure_tensor(f, uv, j/10, t, b, .5), rtol=1e-6)
    r = Rotation.from_euler('xyz', [30, 20, 40], degrees=True).as_matrix()
    z = structure_tensor(f, uv, j@r.T, t@r.T, b@r.T, .05)
    # The existing gradient cache stores float32 world vectors.
    np.testing.assert_allclose(a, z, rtol=1e-6, atol=3e-6)


def test_constant_isotropic_and_sliver_are_exact_covariance_identities():
    f, uv, j, t, b = stripe_tensor()
    constant = TextureFeatures(np.full((32, 32, 3), 128, np.uint8))
    zero = structure_tensor(constant, uv, j, t, b, .05)
    sigma = np.array([[.04, .02]])
    for tensor in (zero, np.eye(2)[None]*100):
        C, gate = guided_covariance(sigma, tensor, .05)
        assert gate[0] == 0
        np.testing.assert_array_equal(C, np.diag(sigma[0]**2)[None])
    _, gate = guided_covariance(np.array([[.04, 1e-8]]), np.diag([10000., 0.])[None], .05)
    assert gate[0] == 0


def test_boundary_projection_and_normal_thickness():
    C, _ = guided_covariance(np.array([[.04, .04]]), np.diag([10000., 0.])[None], .05, 1)
    m = np.array([[[0., 1.], [1., 0.]]]); limit = np.array([[.03, .03]])
    C = project(C, m, limit)
    assert np.all(np.einsum('nki,nij,nkj->nk', m, C, m) <= limit**2*(1+1e-12))
    scales, q = covariance_parameters(C, np.eye(3)[None], np.array([.0008]))
    np.testing.assert_array_equal(scales[:, 2], [.0008])
    r = Rotation.from_quat(q, scalar_first=True).as_matrix()
    np.testing.assert_allclose(r[:, :, 2], [[0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose(r[:, :2, :2]@np.array([np.diag(scales[0, :2]**2)])@r[:, :2, :2].transpose(0,2,1), C, atol=1e-12)


def test_boundary_query_with_fewer_segments_than_requested_and_closed_chart():
    from mesh_to_gs.support_constraints import constraints
    p = np.array([[0., .1, 0.]])
    n, t, b = np.array([[0., 0, 1]]), np.array([[1., 0, 0]]), np.array([[0., 1, 0]])
    edges = (np.array([[0., 0, 0]]), np.array([[.01, 0, 0]]), np.array([0]), np.array([False]))
    m, limit = constraints(p, n, t, b, np.array([0]), edges, np.array([.2]))
    assert m.shape == (1, 8, 2) and np.isfinite(limit).sum() == 1
    empty = tuple(a[:0] for a in edges)
    m, limit = constraints(p, n, t, b, np.array([0]), empty, np.array([.2]))
    assert not m.any() and np.isinf(limit).all()


def test_chart_local_boundary_query_cannot_be_starved_by_foreign_charts():
    from mesh_to_gs.support_constraints import constraints
    # 40 foreign edges are closer than the one true boundary of chart 100.
    y = np.r_[np.linspace(.001,.04,40), .1]
    a = np.column_stack((np.zeros(41), y, np.zeros(41)))
    b = a+np.array([.01,0,0])
    owner = np.r_[np.zeros(40, int), 100]
    edges = (a,b,owner,np.zeros(41,bool))
    p = np.array([[.005,0,0.]])
    n,t,bt = np.array([[0.,0,1]]),np.array([[1.,0,0]]),np.array([[0.,1,0]])
    _, old = constraints(p,n,t,bt,np.array([100]),edges,np.array([.2]))
    assert np.isinf(old).all()
    m, limit = constraints(p,n,t,bt,np.array([100]),edges,np.array([.2]),chart_local=True)
    assert np.isfinite(limit).sum() == 1
    np.testing.assert_allclose(limit[np.isfinite(limit)], [.05], rtol=1e-6)
    np.testing.assert_allclose(np.abs(m[0,0]), [0,1])


def test_audit_seam_classification_requires_complete_geometry():
    from mesh_to_gs.surface import coplanar_seam_edges
    from mesh_to_gs.support_constraints import chart_boundaries, constraints
    triangles = np.array([[[0.,0,0],[1,0,0],[0,1,0]], [[1.,0,0],[1,1,0],[0,1,0]]])
    chart = np.array([8,9]); normals = np.tile([0.,0,1], (2,1))
    full = coplanar_seam_edges(triangles, chart, normals)
    assert full[0,1]
    partial = coplanar_seam_edges(triangles[:1], chart[:1], normals[:1])
    assert not partial.any()
    args = (np.array([[.4,.4,0]]), normals[:1], np.array([[1.,0,0]]), np.array([[0.,1,0]]), chart[:1])
    limits = []
    for mask in (partial, full[:1]):
        edges = chart_boundaries(triangles[:1], chart[:1], mask)
        _, limit = constraints(*args, edges, np.array([.3]), multiple=2, seam_multiple=1,
                               piece=.1, chart_local=True)
        limits.append(limit[np.isfinite(limit)][0])
    np.testing.assert_allclose(limits[1], 2*limits[0], rtol=1e-6)
