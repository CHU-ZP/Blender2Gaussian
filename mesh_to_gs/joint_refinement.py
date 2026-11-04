"""Joint image/Mesh objectives with moving and splitting surface Gaussians.

SH0 and a normal thickness of 2% of the smaller tangent scale are retained.
Normals stay fixed. Neighbours are rebuilt from current centres and after
every split; local residuals are recomputed inside autograd.
"""
import math
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F


def quaternion_product(a, b):
    w, x, y, z = a.unbind(-1)
    v, i, j, k = b.unbind(-1)
    return torch.stack((w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                        w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v), -1)


def quaternion_matrix(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def render_loss(image, target, dssim_weight=.2, window=11):
    """Original 3DGS objective: (1-lambda)*L1 + lambda*(1-SSIM).

    Inputs are HWC sRGB images, including the background. SSIM uses the usual
    11x11 Gaussian window (sigma=1.5), C1=.01^2, C2=.03^2 and zero padding.
    """
    a, b = image.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
    axis = torch.arange(window, dtype=a.dtype, device=a.device) - window // 2
    g = torch.exp(-axis.square() / (2*1.5**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :])[None, None].expand(3, 1, -1, -1).contiguous()
    # Pool the five moments together in one convolution.
    moments = torch.cat((a, b, a*a, b*b, a*b), 0)
    mu_a, mu_b, e_a2, e_b2, e_ab = F.conv2d(moments, kernel, padding=window//2, groups=3).unbind(0)
    va, vb, cov = e_a2-mu_a.square(), e_b2-mu_b.square(), e_ab-mu_a*mu_b
    ssim = (((2*mu_a*mu_b+.01**2)*(2*cov+.03**2)) /
            ((mu_a.square()+mu_b.square()+.01**2)*(va+vb+.03**2))).mean()
    l1 = (image-target).abs().mean()
    return (1-dssim_weight)*l1 + dssim_weight*(1-ssim), l1, ssim


class MovingGaussians(nn.Module):
    def __init__(self, positions, log_scales, quats, rgb, opacity_logits, faces,
                 scale_factor=2., max_displacement=4.):
        super().__init__()
        self.position = nn.Parameter(positions.clone())
        self.log_tangent = nn.Parameter(log_scales[:, :2].clone())
        self.rotation = nn.Parameter(torch.zeros((len(positions), 1),
                                                device=positions.device, dtype=positions.dtype))
        self.rgb = nn.Parameter(rgb.clone())
        self.opacity_logits = nn.Parameter(opacity_logits.clone())
        self.register_buffer('q_ref', F.normalize(quats, dim=-1))
        self.register_buffer('position_ref', positions.clone())
        self.register_buffer('log_ref', log_scales.clone())
        self.register_buffer('rgb_ref', rgb.clone())
        self.register_buffer('attachment_point', positions.clone())
        self.register_buffer('attachment_normal', quaternion_matrix(self.q_ref)[..., 2].contiguous())
        self.register_buffer('faces', faces.to(torch.int32))
        self.register_buffer('roots', torch.arange(len(positions), device=positions.device))
        self.register_buffer('frozen', (log_scales[:, 1] < math.log(1e-6)) |
                             (log_scales[:, 1]-log_scales[:, 0] < math.log(1e-4)))
        self.max_log, self.max_displacement = math.log(scale_factor), max_displacement

    def attachment_loss(self):
        """Dense tangent-plane distance; exact sampled targets update the planes.

        Every centre receives geometry gradients each step. Sampling exact
        queries alone would leave most of a multi-million-row model without
        a centre constraint between sparse visits. Initial planes come from
        owning Mesh faces and are updated by complete-Mesh nearest queries.
        """
        signed = ((self.position-self.attachment_point)*self.attachment_normal).sum(-1)
        unit = self.log_ref[:, :2].exp().max(-1).values.detach().clamp(min=.001)
        return (signed/unit).square().mean()

    def support(self, ids=None):
        ids = slice(None) if ids is None else ids
        rotation = torch.where(self.frozen[ids, None], 0., self.rotation[ids])
        half = rotation/2
        delta_q = torch.cat((half.cos(), torch.zeros_like(half).expand(-1, 2), half.sin()), -1)
        q = quaternion_product(self.q_ref[ids], delta_q)
        tangent = torch.where(self.frozen[ids, None], self.log_ref[ids, :2], self.log_tangent[ids]).exp()
        normal = torch.where(self.frozen[ids], self.log_ref[ids, 2].exp(), .02*tangent.min(-1).values)
        return q, torch.cat((tangent, normal[:, None]), -1)

    def mesh_losses(self, points, ids, colours, surface_count, tolerance=1e-4, empty_weight=1.):
        q, scales = self.support(ids.reshape(-1))
        rotation = quaternion_matrix(q).reshape(*ids.shape, 3, 3)
        residual = points[:, None]-self.position[ids]
        local = torch.einsum('nki,nkij->nkj', residual, rotation)
        scales = scales.reshape(*ids.shape, 3).clamp(min=1e-7)
        normal = torch.sqrt(scales[..., 2].square()+tolerance**2)
        effective = torch.cat((scales[..., :2], normal[..., None]), -1)
        energy = (local/effective).clamp(-100, 100).square().sum(-1)
        u = self.opacity_logits[ids].sigmoid()*torch.exp(-.5*energy)
        occupancy = -torch.expm1(torch.log1p(-u.clamp(max=1-1e-6)).sum(-1))
        colour = (u[..., None]*self.rgb[ids]).sum(-2)/u.sum(-1).clamp(min=1e-8)[..., None]
        covered = (1-occupancy[:surface_count]).square().mean()
        empty = occupancy[surface_count:].square().mean()
        appearance = (colour[:surface_count]-colours).square().mean()
        return dict(geometry=covered+empty_weight*empty, surface=covered, empty=empty, appearance=appearance)

    @torch.no_grad()
    def clamp_(self):
        self.log_tangent.copy_(torch.maximum(torch.minimum(self.log_tangent, self.log_ref[:, :2]+self.max_log),
                                             self.log_ref[:, :2]-self.max_log))
        self.rotation.clamp_(-math.pi/4, math.pi/4)
        radius = self.max_displacement*self.log_ref[:, :2].exp().max(-1).values.clamp(min=1e-6)
        offset = self.position-self.position_ref
        self.position.copy_(self.position_ref+offset*(radius/offset.norm(dim=-1).clamp(min=1e-12)).clamp(max=1)[:, None])
        self.rgb.clamp_(0, 1)
        self.opacity_logits.clamp_(-10, 10)

    @torch.no_grad()
    def split(self, parents, optimizer):
        """Replace each parent by two children; preserve Adam state for surviving rows."""
        if not len(parents):
            return
        device = self.position.device
        keep = torch.ones(len(self.position), dtype=torch.bool, device=device)
        keep[parents] = False
        surviving = torch.where(keep)[0]
        sources = torch.cat((surviving, parents.repeat_interleave(2)))
        child_start = len(surviving)
        q, scales = self.support(parents)
        rotation = quaternion_matrix(q)
        major = scales[:, :2].argmax(-1)
        direction = rotation[torch.arange(len(parents), device=device), :, major]
        offset = .5*scales[:, :2].max(-1).values[:, None]*direction
        child_position = (self.position[parents, None]+offset[:, None]*
                          torch.tensor([-1., 1.], device=device)[None, :, None]).reshape(-1, 3)
        for name, old in list(self.named_parameters()):
            values = old[sources].clone()
            if name == 'position':
                values[child_start:] = child_position
            elif name == 'log_tangent':
                values[child_start:] -= math.log(1.6)
            new = nn.Parameter(values, requires_grad=old.requires_grad)
            for group in optimizer.param_groups:
                group['params'] = [new if p is old else p for p in group['params']]
            state = optimizer.state.pop(old, {})
            remapped = {}
            for key, value in state.items():
                if torch.is_tensor(value) and value.shape == old.shape:
                    value = value[sources].clone()
                    value[child_start:] = 0
                remapped[key] = value
            if remapped:
                optimizer.state[new] = remapped
            setattr(self, name, new)
        for name, value in list(self.named_buffers()):
            mapped = value[sources].clone()
            setattr(self, name, mapped)
        self.log_ref[child_start:, :2] = self.log_tangent[child_start:]
        self.log_ref[child_start:, 2] = (.02*self.log_tangent[child_start:].exp().min(-1).values).log()


class ScreenGradientSplits:
    def __init__(self, model):
        self.reset(model)

    def reset(self, model):
        self.gradient = torch.zeros(len(model.position), device=model.position.device, dtype=model.position.dtype)
        self.count = torch.zeros_like(self.gradient)

    @torch.no_grad()
    def accumulate(self, info):
        grad = info['means2d'].grad
        if grad is None:
            raise RuntimeError('Renderer did not retain projected-centre gradients')
        grad = grad.clone()
        grad[:, 0] *= info['width']/2
        grad[:, 1] *= info['height']/2
        ids = info['gaussian_ids']
        self.gradient.index_add_(0, ids, grad.norm(dim=-1))
        self.count.index_add_(0, ids, torch.ones_like(ids, dtype=grad.dtype))

    @torch.no_grad()
    def choose(self, model, budget=None, threshold=2e-4):
        score = self.gradient/self.count.clamp(min=1)
        eligible = (~model.frozen) & (self.count > 0) & (score >= threshold)
        candidates = torch.where(eligible)[0]
        if budget is None:
            return candidates
        n = min(budget, len(candidates))
        return candidates[torch.topk(score[candidates], n).indices] if n else candidates[:0]


class DynamicNeighbours:
    """Periodic current-centre BVH equivalent via CPU kD-tree, outside autograd.

    K contributors selected from nearest 8*K with a spherical support bound.
    This bound permits rotation in all directions. It remains a finite-candidate
    approximation; the age, rebuild timings and wider-neighbour audit are saved.
    """
    def __init__(self, points, k=32):
        self.points = np.asarray(points, np.float32)
        self.k = k
        self.history = []

    @torch.no_grad()
    def rebuild(self, model, step):
        start = time.perf_counter()
        means = model.position.detach().cpu().numpy().astype(np.float64)
        _, scales = model.support()
        radius = scales[:, :2].max(-1).values.cpu().numpy().clip(1e-7)
        tree = cKDTree(means)
        n = min(self.k, len(means))
        result = np.empty((len(self.points), n), np.int64)
        shortlist = min(8*self.k, len(means))
        for a in range(0, len(self.points), 2048):
            points = self.points[a:a+2048]
            distance, ids = tree.query(points, k=shortlist, workers=8)
            ids, distance = ids.reshape(len(points), -1), distance.reshape(len(points), -1)
            # 2x current radius is a conservative envelope for moderate changes
            # between rebuilds, including normal tilt. No stale local residuals.
            score = (distance/(2*radius[ids]))**2
            selected = np.argpartition(score, n-1, axis=1)[:, :n]
            result[a:a+len(points)] = np.take_along_axis(ids, selected, axis=1)
        self.ids = torch.as_tensor(result, device=model.position.device)
        self.step = step
        self.history.append(dict(step=step, gaussians=len(means), seconds=time.perf_counter()-start,
                                 neighbours=n, shortlist=shortlist))
        return self.history[-1]

    @torch.no_grad()
    def audit(self, model, surface_count, rows=256):
        """Compare K with up to 16*K Euclidean neighbours at the final state."""
        means = model.position.detach().cpu().numpy().astype(np.float64)
        chosen = np.linspace(0, len(self.points)-1, min(rows, len(self.points)), dtype=int)
        _, wide = cKDTree(means).query(self.points[chosen], k=min(16*self.k, len(means)), workers=8)
        wide = torch.as_tensor(wide.reshape(len(chosen), -1), device=model.position.device)
        points = torch.as_tensor(self.points[chosen], device=model.position.device)
        def occupancy(ids):
            q, s = model.support(ids.reshape(-1))
            r = quaternion_matrix(q).reshape(*ids.shape, 3, 3)
            local = torch.einsum('nki,nkij->nkj', points[:, None]-model.position[ids], r)
            s = s.reshape(*ids.shape, 3).clamp(min=1e-7)
            s[..., 2] = torch.sqrt(s[..., 2].square()+1e-8)
            u = model.opacity_logits[ids].sigmoid()*torch.exp(-.5*(local/s).clamp(-100, 100).square().sum(-1))
            return -torch.expm1(torch.log1p(-u.clamp(max=1-1e-6)).sum(-1))
        error = (occupancy(wide)-occupancy(self.ids[chosen])).abs().cpu().numpy()
        return dict(rows=len(chosen), comparison_k=wide.shape[1], mean_absolute_alpha_difference=float(error.mean()),
                    max_absolute_alpha_difference=float(error.max()),
                    limitation='Finite Euclidean comparison, not an exact all-Gaussian field')
