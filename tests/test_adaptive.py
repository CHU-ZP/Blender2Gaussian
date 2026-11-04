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
