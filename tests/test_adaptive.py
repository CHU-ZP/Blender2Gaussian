"""World-space texture/curvature importance and fixed-budget contracts."""
import numpy as np
from mesh_to_gs.surface import charts,triangle_strata,allocate
from mesh_to_gs.adaptive import TextureFeatures,uv_jacobians,prepare_importance


def patch(scale=1):
    t=np.array([[[0.,0,0],[1,0,0],[0,1,0]],[[1,0,0],[1,1,0],[0,1,0]]])*scale
    return charts(t,np.array([[0,1,2],[1,3,2]]),t[:,:,:2]/scale)


def test_uv_stretch_is_converted_to_world_gradient():
    x=(np.arange(128)+.5)/128
    image=np.tile((x*255).astype('uint8')[None,:,None],(128,1,3))
    f=TextureFeatures(image)
    a,b=patch(),patch(10)
    ja,jb=uv_jacobians(a),uv_jacobians(b)
    np.testing.assert_allclose(ja,jb*10,atol=1e-12)
    uv=np.array([[.4,.5],[.6,.5]])
    ga=f.gradients(uv,ja,.001);gb=f.gradients(uv,jb,.001)
    np.testing.assert_allclose(ga,gb*10,rtol=1e-5,atol=1e-6)
    assert np.all(a.curvature==0)


def test_constant_texture_has_no_importance_and_exact_budget():
    s=patch();f=TextureFeatures(np.full((64,64,3),128,np.uint8))
    w=prepare_importance(s,f,.02)
    np.testing.assert_allclose(w,1)
    assert allocate(s.areas*w,999).sum()==999


def test_adaptive_subdivision_moves_points_to_detail_without_leaving_surface():
    t=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    # High importance on a small physical region near vertex 1.
    field=lambda b:1+7*(b[:,1]>.5)
    a=triangle_strata(4096,np.random.default_rng(5),t)
    b=triangle_strata(4096,np.random.default_rng(5),t,field)
    assert (b[:,1]>.5).mean()>(a[:,1]>.5).mean()+.3
    assert len(b)==4096 and b.min()>=0
    np.testing.assert_allclose(b.sum(1),1)


def test_curvature_scales_with_world_geometry():
    phi=np.linspace(0,np.pi/2,25)
    xyz=np.array([[np.cos(a),np.sin(a),z] for a in phi for z in (0.,1.)])
    ids=[]
    for i in range(24):
        a=2*i;ids.extend([[a,a+2,a+1],[a+2,a+3,a+1]])
    ids=np.array(ids)
    a=charts(xyz[ids],ids);b=charts(xyz[ids]*10,ids)
    assert np.median(a.curvature)>.5
    np.testing.assert_allclose(a.curvature,b.curvature*10,rtol=1e-10)


def test_thin_texture_builds_mips_in_the_long_direction():
    image=np.zeros((1,256,3),np.uint8);image[:,::2]=255
    f=TextureFeatures(image)
    assert len(f.levels)==8
    np.testing.assert_allclose(f.levels[-1][0],0)


def test_large_face_quadrature_detects_periodic_texture_edges():
    yy,xx=np.mgrid[:512,:512]
    image=np.repeat(((((xx*32//512)+(yy*32//512))%2)*255).astype(np.uint8)[...,None],3,axis=2)
    f=TextureFeatures(image);s=patch(2)
    w=prepare_importance(s,f,.018,texture_weight=4)
    assert np.min(w)>1.2


def test_narrow_surface_density_tracks_length_instead_of_width():
    from mesh_to_gs.adaptive import narrow_importance
    def mass(width):
        v=np.array([[0.,0,0],[10,0,0],[0,width,0],[10,width,0]])
        ids=np.array([[0,1,2],[1,3,2]])
        s=charts(v[ids],ids)
        return s.component_areas*(1+narrow_importance(s,.02,support=True))
    np.testing.assert_allclose(mass(.005),mass(.01),rtol=.001)


def test_directional_support_detects_gap_between_dense_rows():
    from mesh_to_gs.surface import covariance
    from scipy.spatial.transform import Rotation
    x,y=np.meshgrid(np.arange(-5,6)*.10,np.arange(-50,51)*.01)
    p=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)));n=np.tile([0.,0,1],(len(p),1))
    sc,q,_,area=covariance(p,n,np.empty((0,2,3)),np.sqrt(.001),return_areas=True,directional_support=True)
    R=Rotation.from_quat(q,scalar_first=True).as_matrix();C=np.einsum('nik,nk,njk->nij',R,sc**2,R)
    central=(np.abs(p[:,0])<.25)&(np.abs(p[:,1])<.3)
    # The sparse x direction needs greater support, although nearest samples
    # are predominantly along y. A covariance of nearest deltas gets it wrong.
    assert np.median(C[central,0,0])>2*np.median(C[central,1,1])
