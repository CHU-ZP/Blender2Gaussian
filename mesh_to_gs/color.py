"""sRGB transfer functions shared by material sampling and render validation."""
import numpy as np


# Exposure for scene-linear PBR references and legacy radiance predictions.
# Display-linear Gaussian appearance must not receive this transform again.
MATERIAL_EXPOSURE_STOPS = 3.0


def _where(values):
    if isinstance(values,np.ndarray):
        return np.where
    import torch
    return torch.where


def srgb_to_linear(values):
    return _where(values)(values<=.04045,values/12.92,((values+.055)/1.055).clip(0)**2.4)


def linear_to_srgb(values):
    # Keep the unselected power branch differentiable at black as well: torch
    # evaluates both branches of where, and 0**(1/2.4) has an infinite derivative.
    return _where(values)(values<=.0031308,values*12.92,1.055*values.clip(.0031308, None)**(1/2.4)-.055)


def composite_display_rgb(rgb, alpha, *, display_transform='standard', exposure_stops=0.):
    """White-film sRGB from premultiplied linear RGB (NumPy or Torch).

    Standard takes display-linear baked appearance. ACES takes scene-linear
    radiance; apply it only to radiance models, never to baked display colors.
    """
    coverage = alpha.clip(0, 1)
    if display_transform == 'aces_fitted':
        radiance = rgb / coverage.clip(1e-6, None)
        rgb = aces_fitted(radiance * (2. ** exposure_stops)) * coverage
    elif display_transform != 'standard':
        raise ValueError(f'Unknown display transform: {display_transform}')
    return linear_to_srgb((rgb + 1 - coverage).clip(0, 1)).clip(0, 1)


def aces_fitted(values):
    """Compress scene-linear radiance into display-linear RGB before sRGB.

    This fitted ACES curve keeps bright PBR highlights and white materials
    below hard clipping while preserving the low end. It accepts NumPy arrays
    and Torch tensors so references and Gaussian predictions share a transform.
    """
    if isinstance(values, np.ndarray):
        values = np.maximum(values, 0)
        mapped = values*(2.51*values+.03)/(values*(2.43*values+.59)+.14)
        return np.clip(mapped, 0, 1)
    values = values.clamp(min=0)
    mapped = values*(2.51*values+.03)/(values*(2.43*values+.59)+.14)
    return mapped.clamp(0, 1)
