"""Real rasterizer acceptance: opaque coverage and depth on an unlit plane."""
import numpy as np
import pytest


def test_stratified_plane_coverage_and_depth():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from scripts.render_gaussians import configure_cuda
    configure_cuda()
    from gsplat import rasterization
    from mesh_to_gs.surface import charts,sample_component,covariance
    tri=np.array([[[0,0,3],[1,0,3],[0,1,3]],[[1,0,3],[1,1,3],[0,1,3]]],float)
    s=charts(tri,np.array([[0,1,2],[1,3,2]]))
    p,n,_,edges,h,_=sample_component(s,0,4096,123)
    scales,q,_=covariance(p,n,edges,h)
    def cuda(a):return torch.tensor(a,dtype=torch.float32,device='cuda')
    with torch.no_grad():
        image,alpha,_=rasterization(means=cuda(p),quats=cuda(q),scales=cuda(scales),
            opacities=cuda(np.full(len(p),.99)),colors=cuda(np.full((len(p),3),.5)),
            viewmats=cuda(np.eye(4)[None]),Ks=cuda([[[600,0,156],[0,600,156],[0,0,1]]]),
            width=512,height=512,render_mode='RGB+ED',packed=True,eps2d=.3)
    # Plane projects to [156,356]^2. Exclude its 8px antialiasing/boundary band.
    coverage=alpha[0,164:348,164:348,0].cpu().numpy()
    depth=image[0,164:348,164:348,3].cpu().numpy()
    assert (coverage>=.95).mean()>.99
    assert coverage.mean()>.99
    np.testing.assert_allclose(depth,3,atol=2e-5)


def test_directional_support_covers_uneven_rows():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from scripts.render_gaussians import configure_cuda
    configure_cuda()
    from gsplat import rasterization
    from mesh_to_gs.surface import covariance
    columns=np.arange(-15,16)
    x,y=np.meshgrid(columns*.1+.025*np.where(columns%2,1.,-1.),np.arange(-150,151)*.01)
    p=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,3.)))
    n=np.tile([0.,0,1],(len(p),1))
    sc,q,_,area=covariance(p,n,np.empty((0,2,3)),np.sqrt(.001),overlap=.7,
        directional_support=True,return_areas=True)
    def cuda(a):return torch.as_tensor(np.asarray(a),dtype=torch.float32,device='cuda')
    with torch.no_grad():
        image,alpha,_=rasterization(means=cuda(p),quats=cuda(q),scales=cuda(sc),
            opacities=cuda(np.full(len(p),.995)),colors=cuda(np.full((len(p),3),.5)),
            viewmats=cuda(np.eye(4)[None]),Ks=cuda([[[500,0,320],[0,500,240],[0,0,1]]]),
            width=640,height=480,render_mode='RGB+ED',packed=True,eps2d=.3)
    # A row may fill 48 nearest slots before any point across the wide gap.
    coverage=alpha[0,80:400,110:530,0].cpu().numpy()
    assert (coverage>=.95).mean()>.999
    assert coverage.mean()>.99
    np.testing.assert_allclose(image[0,80:400,110:530,3].cpu().numpy(),3,atol=2e-5)
