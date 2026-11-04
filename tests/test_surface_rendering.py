"""Real rasterizer acceptance: opaque coverage and depth on an unlit plane."""
import numpy as np
import pytest


def test_cuda_configuration_updates_an_earlier_extension_import(monkeypatch):
    import os
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from torch.utils import cpp_extension
    from scripts.render_gaussians import configure_cuda
    monkeypatch.setattr(cpp_extension, 'CUDA_HOME', '/tmp/stale-cuda-toolkit')
    configure_cuda()
    assert cpp_extension.CUDA_HOME == os.environ['CUDA_HOME']


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
            width=512,height=512,render_mode='RGB+ED',packed=True,eps2d=.05)
    # Plane projects to [156,356]^2. Exclude its 8px antialiasing/boundary band.
    coverage=alpha[0,164:348,164:348,0].cpu().numpy()
    depth=image[0,164:348,164:348,3].cpu().numpy()
    assert (coverage>=.95).mean()>.99
    assert coverage.mean()>.99
    np.testing.assert_allclose(depth,3,atol=2e-5)
