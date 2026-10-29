"""Transfer values and known two-layer depth/compositing contracts."""
import numpy as np
import pytest
from mesh_to_gs.color import srgb_to_linear,linear_to_srgb


def test_linear_black_white_average_and_round_trip():
    values=np.linspace(0,1,101)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(values)),values,atol=1e-12)
    assert abs(float(linear_to_srgb(np.array(.5)))-.735356983)<1e-8
    np.testing.assert_allclose(srgb_to_linear(np.array([0.,1.])),[0,1])


def test_gpu_color_and_two_layer_expected_depth():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from scripts.render_gaussians import configure_cuda
    configure_cuda()
    from gsplat import rasterization
    def gpu(x):return torch.tensor(x,dtype=torch.float32,device='cuda')
    c=gpu([0.,.1,.5,1.])
    torch.testing.assert_close(linear_to_srgb(srgb_to_linear(c)),c)
    with torch.no_grad():
        out,a,_=rasterization(means=gpu([[0,0,2],[0,0,4]]),quats=gpu([[1,0,0,0]]*2),
            scales=gpu([[.2,.2,.001],[.4,.4,.001]]),opacities=gpu([.7,.9]),colors=gpu([[1,0,0],[0,0,1]]),
            viewmats=gpu(np.eye(4)[None]),Ks=gpu([[[1,0,.5],[0,1,.5],[0,0,1]]]),width=1,height=1,packed=True,render_mode='RGB+ED')
    np.testing.assert_allclose(a.cpu(),.97,atol=1e-6)
    np.testing.assert_allclose(out[0,0,0,:3].cpu(),[.7,0,.27],atol=1e-6)
    assert abs(out[0,0,0,3].item()-(.7*2+.27*4)/.97)<1e-6
