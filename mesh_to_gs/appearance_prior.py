"""Training-camera PBR point colours, with conservative depth visibility.

This is a training adaptation of SH0 appearance, not a replacement geometric
initializer. No test cameras or test targets enter the bake or prior.
"""
import torch
from torch.nn import functional as F

from .color import srgb_to_linear


def projected_samples(position, matrix, intrinsics, target, depth, mask, near, far):
    camera = position@matrix[:3, :3].T+matrix[:3, 3]
    z = camera[:, 2]
    pixels = camera@intrinsics.T
    uv = pixels[:, :2]/z.clamp(min=1e-8)[:, None]
    height, width = depth.shape
    valid = (z > near) & (z < far) & (uv[:, 0] >= .5) & (uv[:, 0] < width-.5)
    valid &= (uv[:, 1] >= .5) & (uv[:, 1] < height-.5)
    ids = torch.where(valid)[0]
    if not len(ids):
        return ids, position.new_empty((0, 3))
    selected_uv = uv[ids]
    x, y = selected_uv.floor().long().unbind(-1)
    visible = mask[y, x] & ((z[ids]-depth[y, x]).abs() <= .0025+.001*depth[y, x])
    ids, selected_uv = ids[visible], selected_uv[visible]
    grid = (2*selected_uv/position.new_tensor([width, height])-1)[None, :, None]
    if len(ids):
        samples = F.grid_sample(target, grid, mode='bilinear', padding_mode='border',
                               align_corners=False)[0, :, :, 0].T
    else:
        samples = position.new_empty((0, 3))
    return ids, samples


@torch.no_grad()
def bake_training_appearance(model, targets, depths, masks, matrices, intrinsics, cameras, chunk=262144):
    total = torch.zeros_like(model.rgb)
    square = torch.zeros_like(model.rgb)
    count = torch.zeros(len(model.rgb), device=model.rgb.device, dtype=torch.int32)
    for i, camera in enumerate(cameras):
        target = srgb_to_linear(targets[i]).permute(2, 0, 1).contiguous()[None]
        for start in range(0, len(model.position), chunk):
            stop = min(start+chunk, len(model.position))
            ids, samples = projected_samples(model.position[start:stop], matrices[i][0], intrinsics[i][0],
                         target, depths[i], masks[i], camera['near'], camera['far'])
            if hasattr(model, 'transparent'):
                opaque = ~model.transparent[start+ids]
                ids, samples = ids[opaque], samples[opaque]
            ids = ids+start
            total[ids] += samples
            square[ids] += samples.square()
            count[ids] += 1
    mean = total/count.clamp(min=1)[:, None]
    variance = (square/count.clamp(min=1)[:, None]-mean.square()).clamp(min=0).mean(-1)
    weight = (count >= 2).to(model.rgb.dtype)/(1+variance/.01)
    reference = torch.where((count > 0)[:, None], mean, model.rgb).clamp(0, 1)
    model.register_buffer('baked_rgb', reference)
    model.register_buffer('baked_weight', weight)
    model.register_buffer('baked_observations', count)
    model.rgb.copy_(reference)
    return dict(observed_rows=int((count > 0).sum()), prior_rows=int((count >= 2).sum()),
                total_rows=len(count), observation_mean=float(count.float().mean()),
                mean_variance=float(variance[count >= 2].mean()),
                target_space='Display-linear PBR point appearance from training cameras only',
                visibility='Verified opaque depth mask and camera-Z agreement within 2.5 mm +0.1% depth',
                confidence='At least 2 observations; 1/(1+RGB variance/0.01)',
                unobserved_colour='Weak source chromaticity guard with 0.25 tolerance; luminance unconstrained',
                adaptation='Set observed SH0 colours from training PBR; original input PLY unchanged')


def appearance_prior_loss(model):
    error = (model.rgb-model.baked_rgb).square().mean(-1)
    measured = (error*model.baked_weight).sum()/model.baked_weight.sum().clamp(min=1)
    # Points with no reliable centre observation can still render footprint
    # fragments. Avoid unconstrained saturated colours on those hidden rows,
    # while allowing their brightness to fit illumination and shadows.
    unknown = model.baked_weight == 0
    def chroma(rgb):
        mean = rgb.mean(-1, keepdim=True)
        return (rgb-mean)/mean.clamp(min=.03)
    residual = (chroma(model.rgb)-chroma(model.rgb_ref)).abs()
    guard = (residual-.25).clamp(min=0).square().mean(-1)
    fallback = (guard*unknown).sum()/unknown.sum().clamp(min=1)
    return measured+.25*fallback


def project_hue(rgb, reference, tolerance):
    """Allow source hue to desaturate, but bound changes away from that hue.

    The reference is a measured PBR point colour when reliable, otherwise
    source BaseColor. RGB mean can change freely within the display gamut.
    This is an explicit appearance constraint, not a physical lighting model.
    """
    mean = rgb.mean(-1, keepdim=True)
    ref_mean = reference.mean(-1, keepdim=True)
    chroma = (rgb-mean)/mean.clamp(min=1e-6)
    source = (reference-ref_mean)/ref_mean.clamp(min=1e-6)
    strength = ((chroma*source).sum(-1, keepdim=True)/
                source.square().sum(-1, keepdim=True).clamp(min=1e-12)).clamp(0, 1)
    anchor = strength*source
    residual = chroma-anchor
    fraction = (tolerance/residual.abs().amax(-1, keepdim=True).clamp(min=1e-12)).clamp(max=1)
    projected = anchor+fraction*residual
    # Contract toward neutral if a channel would become negative; reduce
    # mean uniformly for the upper gamut limit, preserving the hue bound.
    projected = projected/(-projected.amin(-1, keepdim=True)).clamp(min=1)
    ratio = 1+projected
    allowed_mean = torch.minimum(mean, 1/ratio.amax(-1, keepdim=True).clamp(min=1))
    return (allowed_mean*ratio).clamp(0, 1)


@torch.no_grad()
def constrain_appearance_hue_(model, tolerance):
    for start in range(0, len(model.rgb), 1048576):
        stop = start+1048576
        reliable = model.baked_weight[start:stop] >= .5
        reference = torch.where(reliable[:, None], model.baked_rgb[start:stop],
                                model.rgb_ref[start:stop])
        projected = project_hue(model.rgb[start:stop], reference, tolerance)
        model.rgb[start:stop].copy_(torch.where(model.transparent[start:stop, None],
                                               model.rgb[start:stop], projected))
