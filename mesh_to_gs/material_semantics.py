"""Verified material metadata and approximate thin-layer alpha conversion.

Material alpha is a sheet property. It is not the opacity of each overlapping
Gaussian. For low-alpha sheets we divide optical depth by the expected overlap
2*pi*sigma_x*sigma_y*density. Boundary, perspective and nonuniform sampling
make this an initialization approximation, not a physical renderer.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .mesh_supervision import digest


def sheet_splat_opacity(alpha, overlap):
    alpha = np.asarray(alpha, np.float64)
    overlap = np.asarray(overlap, np.float64)
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha >= 1)):
        raise ValueError('Thin-sheet alpha must be in [0, 1)')
    if not np.isfinite(overlap).all() or np.any(overlap <= 0):
        raise ValueError('Expected overlap must be finite and positive')
    return -np.expm1(np.log1p(-alpha)/overlap)


def apply_material_semantics(model, directory, input_model, source_hash, cap_factor=2.):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['input_sha256'] != digest(input_model) or manifest['source_sha256'] != source_hash:
        raise ValueError('Material semantics belong to a different initializer/source')
    for name in ('face_alpha.npy', 'face_group.npy', 'group_density.npy', 'row_alpha.npy'):
        if digest(directory/name) != manifest['files'][name]:
            raise ValueError(f'Material semantics file changed: {name}')
    alpha = np.load(directory/'face_alpha.npy', mmap_mode='r')
    face_group = np.load(directory/'face_group.npy', mmap_mode='r')
    density = np.load(directory/'group_density.npy')
    faces = model.faces.cpu().numpy()
    row_alpha = np.asarray(np.load(directory/'row_alpha.npy'))
    if row_alpha.shape != faces.shape:
        raise ValueError('Material alpha must match the original initializer rows')
    transparent = row_alpha < .99
    rows = np.flatnonzero(transparent)
    cap = torch.full_like(model.opacity_logits, 10.)
    with torch.no_grad():
        overlap = 2*np.pi*np.exp(model.log_ref[rows, :2].cpu().numpy()).prod(-1)*density[face_group[faces[rows]]]
        opacity = sheet_splat_opacity(row_alpha[rows], overlap).clip(1/(1+np.exp(10)), 1-1e-6)
        model.opacity_logits[rows] = torch.as_tensor(np.log(opacity/(1-opacity)),
                                               dtype=cap.dtype, device=cap.device)
        maximum = sheet_splat_opacity(1-(1-row_alpha[rows])**cap_factor, overlap).clip(1/(1+np.exp(10)), 1-1e-6)
        cap[rows] = torch.as_tensor(np.log(maximum/(1-maximum)), dtype=cap.dtype, device=cap.device)
    model.register_buffer('material_opacity_cap', cap)
    model.register_buffer('transparent', torch.as_tensor(transparent, device=cap.device))
    return dict(manifest=str(directory/'manifest.json'), manifest_sha256=digest(directory/'manifest.json'),
                transparent_rows=len(rows), cap_optical_depth_factor=cap_factor,
                opacity_quantiles=np.quantile(opacity, [0, .5, .9, 1]).tolist() if len(rows) else [],
                conversion='Sheet optical depth divided by expected Gaussian overlap; approximate',
                initializer_adaptation='In-memory transparent opacity calibration; raw input PLY unchanged')
