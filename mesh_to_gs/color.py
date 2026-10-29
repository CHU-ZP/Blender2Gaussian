"""sRGB transfer functions shared by material sampling and render validation."""
import numpy as np


def _where(values):
    if isinstance(values,np.ndarray):
        return np.where
    import torch
    return torch.where


def srgb_to_linear(values):
    return _where(values)(values<=.04045,values/12.92,((values+.055)/1.055).clip(0)**2.4)


def linear_to_srgb(values):
    return _where(values)(values<=.0031308,values*12.92,1.055*values.clip(0)**(1/2.4)-.055)
