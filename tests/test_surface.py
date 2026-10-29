"""Sampling, topology and covariance invariants for unlit initialization."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mesh_to_gs.surface import (allocate, charts, triangle_strata, sample_component, covariance,
                               small_single_face_components, singleton_components, texture_colors)


def square(subdivisions=1, uv=True):
    xy = np.mgrid[0:subdivisions+1, 0:subdivisions+1].reshape(2,-1).T / subdivisions
    vertices = np.column_stack((xy, np.zeros(len(xy))))
    ids=[]
    for i in range(subdivisions):
        for j in range(subdivisions):
            a=i*(subdivisions+1)+j; b=a+subdivisions+1
            ids.extend([[a,b,a+1],[b,b+1,a+1]])
    ids=np.array(ids)
    return charts(vertices[ids],ids,vertices[ids][:,:,:2] if uv else None)


def test_retriangulation_has_same_chart_area_and_budget():
    a,b=square(),square(12)
    np.testing.assert_allclose(a.component_areas,b.component_areas)
    np.testing.assert_allclose(a.perimeter,b.perimeter)
    for s in (a,b):
        assert allocate(s.component_areas,500,minimum=True).tolist()==[500]
        p,*_=sample_component(s,0,500,123)
        assert len(p)==500
        assert np.all((p>=0)&(p<=1))


def test_strata_are_unique_inside_triangle_and_repeatable():
    for count in [1,2,7,128,65536]:
        a=triangle_strata(count,np.random.default_rng(5))
        b=triangle_strata(count,np.random.default_rng(5))
        np.testing.assert_array_equal(a,b)
        assert len(np.unique(a,axis=0))==count
        assert a.min()>=0
        np.testing.assert_allclose(a.sum(1),1)


def test_physical_strata_do_not_inherit_long_triangle_rows():
    from scipy.spatial import cKDTree
    triangle=np.array([[0.,0,0],[10.,0,0],[0,.1,0]])
    bary=triangle_strata(1024,np.random.default_rng(42),triangle)
    p=bary@triangle
    h=np.sqrt(.5/1024)
    distances,_=cKDTree(p).query(p,k=2)
    assert np.median(distances[:,1])>.35*h
    assert np.quantile(distances[:,1],.95)<2*h
    assert bary.min()>0 and len(np.unique(p,axis=0))==1024
    np.testing.assert_allclose(bary.sum(1),1)


def test_uv_seam_and_crease_are_chart_boundaries():
    tri=np.array([[[0,0,0],[1,0,0],[0,1,0]],[[1,0,0],[1,1,0],[0,1,0]]],float)
    ids=np.array([[0,1,2],[1,3,2]])
    uv=tri[:,:,:2].copy()
    assert len(charts(tri,ids,uv).component_areas)==1
    uv[1]+=1
    assert len(charts(tri,ids,uv).component_areas)==2
    tri[1,1,2]=1
    assert len(charts(tri,ids).component_areas)==2
    # Exact duplicate vertices can be welded; close parallel sheets cannot.
    duplicate=np.arange(6).reshape(2,3)
    tri[1,1,2]=0
    assert len(charts(tri,duplicate).component_areas)==1
    assert len(charts(np.concatenate((tri,tri+[0,0,1e-6])),np.arange(12).reshape(4,3)).component_areas)==2


def test_boundary_covariance_and_local_to_world_rotation():
    s=square(8)
    p,n,_,edges,h,_=sample_component(s,0,4096,3)
    sigma,q,capped=covariance(p,n,edges,h)
    R=Rotation.from_quat(q,scalar_first=True).as_matrix()
    np.testing.assert_allclose(R[:,:,2],n,atol=4e-7)
    np.testing.assert_allclose(np.linalg.det(R),1,atol=4e-7)
    assert capped>0 and np.all(sigma>0)
    C=np.einsum('nik,nk,njk->nij',R,sigma**2,R)
    for axis in (0,1):
        # Every tangent 2-sigma footprint stays within the square.
        distance=np.minimum(p[:,axis],1-p[:,axis])
        assert np.max(2*np.sqrt(C[:,axis,axis])-distance)<1e-6
    assert np.median(sigma[:,0]) > .5*h


def test_linear_texture_filtering():
    image=np.array([[[0,0,0],[255,255,255]]],dtype=np.uint8)
    color=texture_colors(image,np.array([[.5,.5],[0,.5],[1,.5]]))
    np.testing.assert_allclose(color,.73535698,atol=1e-6)


@pytest.mark.parametrize('count',[1,2,3])
def test_small_facets_remain_separate_and_inside_their_boundaries(count):
    tri=np.array([[[0,0,0],[1,0,0],[0,1,0]],[[0,0,1e-4],[1,0,1e-4],[0,1,1e-4]]],float)
    s=charts(tri,np.arange(6).reshape(2,3))
    p,n,_,sc,q,_=small_single_face_components(s,np.array([0,1]),count,123,0)
    np.testing.assert_allclose(p[:count,2],0)
    np.testing.assert_allclose(p[count:,2],1e-4)
    R=Rotation.from_quat(q,scalar_first=True).as_matrix()
    np.testing.assert_allclose(R[:,:,2],n,atol=3e-7)
    C=np.einsum('nik,nk,njk->nij',R,sc**2,R)
    for m,d in [(np.array([1.,0,0]),p[:,0]),(np.array([0.,1,0]),p[:,1]),
                (np.array([1.,1,0])/np.sqrt(2),(1-p[:,0]-p[:,1])/np.sqrt(2))]:
        sigma=np.sqrt(np.einsum('i,nij,j->n',m,C,m))
        assert np.max(2*sigma-d)<1e-6


def test_finite_boundary_does_not_collapse_distant_collinear_points():
    p=np.array([[2.,0,0],[2.,.1,0],[2.1,0,0],[2.1,.1,0]])
    n=np.tile([0.,0,1],(4,1))
    # This boundary ends a metre away; its extended line is irrelevant.
    edges=np.array([[[0.,0,0],[1.,0,0]]])
    a,_,_=covariance(p,n,edges,.1)
    b,_,_=covariance(p,n,np.empty((0,2,3)),.1)
    np.testing.assert_allclose(a,b,rtol=1e-6)


def test_boundary_inset_tracks_spacing_instead_of_triangle_altitude():
    s=square()
    p,_,_,_,h,nb=sample_component(s,0,4096,123)
    boundary=p[-nb:,:2]
    distance=np.minimum(boundary,1-boundary).min(1)
    assert nb>0 and np.max(distance)<=.3*h+1e-12


def test_singleton_batch_uses_connected_chart_area_without_cross_sheet_neighbours():
    s0=square()
    triangles=np.concatenate((s0.triangles,s0.triangles+[0,0,1e-5]))
    s=charts(triangles,np.arange(12).reshape(4,3))
    p,n,_,sc,q,_=singleton_components(s,np.array([0,1]))
    np.testing.assert_allclose(p[:,2],[0,1e-5])
    np.testing.assert_allclose(sc[0],sc[1])
    np.testing.assert_allclose(np.linalg.norm(q,axis=1),1,atol=1e-6)


@pytest.mark.parametrize('scale',[1.,1e-5])
@pytest.mark.parametrize('backend',['cuda','python'])
def test_independent_samples_and_tiny_normal(scale,backend):
    import torch
    if backend=='cuda':
        if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
        from mesh_to_gs import mesh_to_gs_cuda
        convert=lambda v,f:mesh_to_gs_cuda(v,f,R=256,k=2,seed=123)
    else:
        from mesh2gs_python import mesh_to_gaussmodel_improved
        def convert(v,f):
            np.random.seed(123)
            return mesh_to_gaussmodel_improved(v,f,R=256,k=2,color_mode='none',device='cpu',max_sh_degree=0)
    R=Rotation.from_euler('xyz',[.4,-.7,.2]).as_matrix()
    vertices=np.array([[0,0,0],[1,0,0],[.2,1,0]],dtype=np.float32)@R.T*scale
    # 65536 Gaussians for ordinary triangle; 1 for a small but valid triangle.
    a=convert(vertices.astype('float32'),np.array([[0,1,2]],np.int32))
    b=convert(vertices.astype('float32'),np.array([[0,1,2]],np.int32))
    p=a.get_xyz.detach().cpu().numpy()
    np.testing.assert_array_equal(p,b.get_xyz.detach().cpu().numpy())
    assert len(np.unique(p,axis=0))==len(p)
    q=a._rotation.detach().cpu().numpy()
    np.testing.assert_allclose(np.linalg.norm(q,axis=1),1,atol=2e-6)
    rot=Rotation.from_quat(q,scalar_first=True).as_matrix()
    np.testing.assert_allclose(rot[:,:,2],np.broadcast_to(R[:,2],(len(q),3)),atol=2e-6)
