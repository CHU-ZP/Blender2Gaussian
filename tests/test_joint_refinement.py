"""Independent checks of moving geometry, normal DOFs and split optimizer state."""
import numpy as np
import pytest
import torch

from mesh_to_gs.joint_refinement import (DynamicNeighbours, MovingGaussians, ScreenGradientSplits,
                                         quaternion_matrix, render_loss)
from mesh_to_gs.mesh_targets import MeshTargets


def model():
    return MovingGaussians(torch.tensor([[0., 0, 0], [.3, 0, 0]], dtype=torch.float64),
                           torch.tensor([[.1, .05, .001], [.1, .05, .001]], dtype=torch.float64).log(),
                           torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]], dtype=torch.float64),
                           torch.full((2, 3), .3, dtype=torch.float64),
                           torch.full((2,), 4.595, dtype=torch.float64), torch.zeros(2, dtype=torch.int32))


def test_original_render_loss_identical_images_and_nonzero_gradients():
    torch.manual_seed(0)
    target = torch.rand((20, 25, 3), dtype=torch.float64)
    loss, l1, ssim = render_loss(target, target)
    assert abs(float(loss)) < 1e-12 and l1 == 0 and abs(float(ssim)-1) < 1e-12
    image = (target*.9).requires_grad_()
    loss, l1, ssim = render_loss(image, target)
    torch.testing.assert_close(loss, .8*l1+.2*(1-ssim))
    loss.backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0


def test_mesh_field_differentiates_current_centres_not_cached_offsets():
    m = model()
    ids = torch.tensor([[0], [0]])
    points = torch.tensor([[.07, 0, 0], [.07, 0, .05]], dtype=torch.float64)
    colours = torch.full((1, 3), .3, dtype=torch.float64)
    terms = m.mesh_losses(points, ids, colours, 1)
    terms['geometry'].backward()
    assert m.position.grad[0, 0] < 0  # centre moves towards the surface query
    initial = float(terms['geometry'].detach())
    with torch.no_grad():
        m.position[0, 0] += .05
    assert float(m.mesh_losses(points, ids, colours, 1)['geometry'].detach()) < initial


def test_dense_attachment_supervises_centres_outside_the_sampled_subset():
    m = model()
    with torch.no_grad():
        m.position[:, 2] = .001
    m.attachment_loss().backward()
    assert (m.position.grad[:, 2] > 0).all()
    assert (m.position.grad[:, :2] == 0).all()  # tangent movement remains free


def test_in_plane_rotation_preserves_normal_and_thickness():
    m = model()
    with torch.no_grad():
        m.rotation[:, -1] = .3
    q, scales = m.support()
    normal = quaternion_matrix(q)[..., 2]
    torch.testing.assert_close(normal, torch.tensor([[0., 0, 1.]]*2, dtype=torch.float64))
    torch.testing.assert_close(scales[:, 2], .02*scales[:, :2].min(-1).values)



def test_split_replaces_parent_and_preserves_adam_for_survivors():
    m = model()
    optim = torch.optim.Adam(m.parameters(), lr=.01)
    sum(p.square().sum() for p in m.parameters()).backward()
    optim.step()
    old_state = optim.state[m.position]['exp_avg'][1].clone()
    old_step = optim.state[m.position]['step'].clone()
    before = m.position[0].detach().clone()
    m.split(torch.tensor([0]), optim)
    assert len(m.position) == 3
    torch.testing.assert_close(m.position[1:].mean(0), before)
    assert m.roots.tolist() == [1, 0, 0] and m.faces.tolist() == [0, 0, 0]
    torch.testing.assert_close(optim.state[m.position]['exp_avg'][0], old_state)
    assert (optim.state[m.position]['exp_avg'][1:] == 0).all()
    torch.testing.assert_close(optim.state[m.position]['step'], old_step)
    assert all(p in optim.state for p in m.parameters())
    optim.zero_grad()
    sum(p.square().sum() for p in m.parameters()).backward()
    optim.step()  # exercises new state sizes and replacement parameter groups
    assert torch.isfinite(m.position).all()


def test_neighbours_rebuild_after_motion_and_split():
    m = model()
    n = DynamicNeighbours(np.array([[.02, 0, 0]], np.float32), k=1)
    n.rebuild(m, 0)
    assert n.ids.tolist() == [[0]]
    with torch.no_grad():
        m.position[0, 0] = 2
    n.rebuild(m, 1)
    assert n.ids.tolist() == [[1]]
    optim = torch.optim.Adam(m.parameters())
    m.split(torch.tensor([1]), optim)
    n.rebuild(m, 2)
    assert int(n.ids.max()) < 3 and n.history[-1]['gaussians'] == 3


def test_screen_gradient_split_selection_has_a_real_threshold_and_budget():
    m = model()
    splits = ScreenGradientSplits(m)
    splits.gradient[:] = torch.tensor([.0001, .003])
    splits.count[:] = 1
    assert splits.choose(m, 1).tolist() == [1]
    assert splits.choose(m, 1, threshold=.01).numel() == 0
    splits.gradient[:] = .003
    assert splits.choose(m, None).tolist() == [0, 1]


def test_exact_closest_targets_include_skinny_faces_and_other_open_sheets():
    origin = np.array([100., 130., 1.])
    local = np.array([[[8., 8., 0], [8.1, 8., 0], [8., 8.+1e-8, 0]],
                      [[0., 0, .002], [1., 0, .002], [0., 1., .002]]])
    targets = MeshTargets(local+origin, origin, device='cpu')
    p = torch.tensor([[8.02, 8., .0003], [.2, .2, .0021]], dtype=torch.float32)
    closest, normal, face = targets.closest(p, torch.tensor([0, 0], dtype=torch.int32))
    torch.testing.assert_close(closest[:, 2], torch.tensor([0., .002]))
    assert face.tolist() == [0, 1]
    assert torch.isfinite(normal).all()


def test_centre_distance_gradient_moves_toward_mesh():
    targets = MeshTargets(np.array([[[0., 0, 0], [1., 0, 0], [0., 1., 0]]]), [0, 0, 0], device='cpu')
    p = torch.tensor([[.2, .2, .01]], requires_grad=True)
    closest, _, _ = targets.closest(p, torch.tensor([0], dtype=torch.int32))
    (p-closest).square().sum().backward()
    assert p.grad[0, 2] > 0 and not closest.requires_grad


