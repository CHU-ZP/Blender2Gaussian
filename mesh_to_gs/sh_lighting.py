"""Read standard PLY spherical-harmonic colour fields for the Gaussian renderer."""
import numpy as np
import torch


def ply_sh_degree(names):
    rest = {name for name in names if name.startswith('f_rest_')}
    for degree in range(4):
        count = 3*((degree+1)**2-1)
        if rest == {f'f_rest_{i}' for i in range(count)}:
            return degree
    raise ValueError('Expected contiguous standard SH fields for degree 0, 1, 2 or 3')


def unpack_sh(rows):
    degree = ply_sh_degree(rows.dtype.names)
    coeffs = np.empty((len(rows), (degree+1)**2, 3), np.float32)
    coeffs[:, 0] = np.column_stack([rows[f'f_dc_{i}'] for i in range(3)])
    if degree:
        rest = np.column_stack([rows[f'f_rest_{i}'] for i in range(3*((degree+1)**2-1))])
        coeffs[:, 1:] = rest.reshape(len(rows), 3, -1).transpose(0, 2, 1)
    return coeffs


def load_sh(rows, device, chunk=262144):
    degree = ply_sh_degree(rows.dtype.names)
    coeffs = torch.empty((len(rows), (degree+1)**2, 3), dtype=torch.float32, device=device)
    for start in range(0, len(rows), chunk):
        coeffs[start:start+chunk] = torch.from_numpy(unpack_sh(rows[start:start+chunk])).to(device)
    return coeffs, degree
