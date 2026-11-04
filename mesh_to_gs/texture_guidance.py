"""Opt-in, world-space texture guidance for fixed-row initializer experiments."""
import numpy as np
from scipy.spatial.transform import Rotation


def structure_tensor(texture, uv, jacobian, tangent, bitangent, spacing):
    """Integrate RGB gradient outer products over a small physical footprint.

    The tensor is in the supplied orthonormal world tangent frame. UV stretch,
    rotation and texel resolution enter through the world-to-UV Jacobian and
    TextureFeatures' footprint-selected mip, never through raw UV axes.
    """
    spacing = np.broadcast_to(spacing, (len(uv),))
    jacobian = np.broadcast_to(jacobian, (len(uv), 2, 3))
    tensor = np.zeros((len(uv), 2, 2), np.float64)
    weights = 0.
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            offset = .35 * spacing[:, None] * (x * tangent + y * bitangent)
            sample_uv = uv + np.einsum('nij,nj->ni', jacobian, offset)
            gradient = texture.gradients(sample_uv, jacobian, spacing)
            local = np.stack((np.einsum('nci,ni->nc', gradient, tangent),
                              np.einsum('nci,ni->nc', gradient, bitangent)), -1)
            weight = np.exp(-.5 * (x*x + y*y))
            tensor += weight * np.einsum('nci,ncj->nij', local, local) / 3
            weights += weight
    return tensor / weights


def guided_covariance(sigma, tensor, spacing, strength=.5, max_aspect=2.):
    """Weak-gradient axis is long; preserve sigma0*sigma1 before clipping.

    Blend SPD shapes, then normalize their determinant. Contrast and tensor
    coherence gate the change, so constant/isotropic textures are identities.
    Very thin geometric footprints stay fixed rather than becoming texture
    ellipses. Returns covariance in the reference axes and the effective gate.
    """
    if not 0 <= strength <= 1 or max_aspect < 1:
        raise ValueError('strength must be in [0,1], max_aspect must be >= 1')
    sigma = np.asarray(sigma, np.float64)
    spacing = np.broadcast_to(spacing, (len(sigma),))
    values, vectors = np.linalg.eigh(tensor)
    values = np.maximum(values, 0)
    coherence = (values[:, 1] - values[:, 0]) / np.maximum(values.sum(1), 1e-30)
    contrast = spacing * np.sqrt(values.sum(1))
    gate = strength * np.clip((contrast - .04) / .20, 0, 1) * coherence
    frozen = (sigma[:, 1] < 1e-6) | (sigma[:, 0] / sigma[:, 1] > 20)
    gate[frozen] = 0
    reg = (.04 / np.maximum(spacing, 1e-30)) ** 2
    aspect = np.sqrt((values[:, 1] + reg) / (values[:, 0] + reg)).clip(1, max_aspect)
    # Eigenvectors are ascending: index 0 points along weak gradients.
    target = np.einsum('nij,nj,nkj->nik', vectors,
                       np.column_stack((aspect, 1/aspect)), vectors)
    area = sigma[:, 0] * sigma[:, 1]
    reference = np.zeros_like(target)
    reference[:, 0, 0] = sigma[:, 0] / sigma[:, 1]
    reference[:, 1, 1] = sigma[:, 1] / sigma[:, 0]
    shape = (1-gate[:, None, None])*reference + gate[:, None, None]*target
    shape /= np.sqrt(np.maximum(np.linalg.det(shape), 1e-30))[:, None, None]
    result = shape * area[:, None, None]
    identity = gate == 0
    result[identity] = 0
    result[identity, 0, 0] = sigma[identity, 0]**2
    result[identity, 1, 1] = sigma[identity, 1]**2
    return result, gate


def covariance_parameters(covariance, reference_rotation, normal_sigma):
    """Convert tangent covariance to sorted sigmas/wxyz, keeping normal sigma."""
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 1e-30)
    major = vectors[:, :, 1]
    axis0 = np.einsum('nij,nj->ni', reference_rotation[:, :, :2], major)
    normal = reference_rotation[:, :, 2]
    axis1 = np.cross(normal, axis0)
    rotation = np.stack((axis0, axis1, normal), -1)
    return np.column_stack((np.sqrt(values[:, 1]), np.sqrt(values[:, 0]), normal_sigma)), \
        Rotation.from_matrix(rotation).as_quat(scalar_first=True)
