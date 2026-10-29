"""Physical scale and export invariants, independent of point sampling order."""

import importlib
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mesh2gs_python as reference


@pytest.fixture(params=["python", "cuda"])
def convert(request):
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU unavailable")
        from mesh_to_gs import mesh_to_gs_cuda
        return mesh_to_gs_cuda

    def cpu(vertices, faces, **kwargs):
        return reference.mesh_to_gaussmodel_improved(
            vertices, faces, device="cpu", color_mode="none", max_sh_degree=0, **kwargs)
    return cpu


def triangle_model(convert, vertices, **kwargs):
    return convert(np.asarray(vertices, np.float32), np.array([[0, 1, 2]], np.int32), R=512, k=2, **kwargs)


def test_unclamped_footprint_uses_actual_allocation(convert):
    vertices = np.array([[0, 0, 0], [.01, 0, 0], [.003, .008, 0]], np.float32)
    model = triangle_model(convert, vertices, min_scale=1e-12, max_scale=1)
    area = 0.5 * np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
    scales = model.get_scaling.detach().cpu().numpy()
    assert len(scales) == 21  # k*A*R^2 is non-integer, approximately 20.97.
    np.testing.assert_allclose(np.pi * scales[:, 0] * scales[:, 1], area / len(scales), rtol=2e-6)


def test_same_allocation_shrinks_with_geometry(convert):
    vertices = np.array([[0, 0, 0], [.0005, 0, 0], [.00015, .0004, 0]], np.float32)
    large = triangle_model(convert, vertices)
    small = triangle_model(convert, vertices * .01)
    assert len(large.get_xyz) == len(small.get_xyz) == 1
    torch.testing.assert_close(small.get_scaling, large.get_scaling * .01, rtol=3e-6, atol=1e-12)


@pytest.mark.parametrize("height", [1e-7, 1e-9])
def test_minimum_scale_and_thickness_respect_sliver_support(convert, height):
    vertices = np.array([[0, 0, 0], [.1, 0, 0], [.03, height, 0]], np.float32)
    model = triangle_model(convert, vertices)
    scales = model.get_scaling.detach().cpu().numpy()
    assert np.isfinite(scales).all() and (scales > 0).all()
    area = 0.5 * np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
    longest = max(np.linalg.norm(vertices[1] - vertices[0]), np.linalg.norm(vertices[2] - vertices[1]),
                  np.linalg.norm(vertices[0] - vertices[2]))
    assert np.all(scales[:, 0] <= .5 * longest / math.sqrt(len(scales)) * (1 + 2e-6))
    assert np.all(scales[:, 1] <= 2 * area / longest / math.sqrt(len(scales)) * (1 + 2e-6))
    assert np.all(scales[:, 2] <= .1 * np.minimum(scales[:, 0], scales[:, 1]) * (1 + 2e-6))


@pytest.mark.parametrize("module_name,converter_name", [
    ("mesh2gs", "mesh_to_gs_cuda"), ("mesh2gs_python", "mesh_to_gaussmodel_improved")])
@pytest.mark.parametrize("preserve", [False, True])
def test_cli_restores_centers_and_standard_deviations(monkeypatch, tmp_path, module_name, converter_name, preserve):
    module = importlib.import_module(module_name)
    vertices = np.array([[100., 130., 0.], [122., 130., 0.], [100., 146., 0.]])
    mesh = SimpleNamespace(vertices=vertices, faces=np.array([[0, 1, 2]]), visual=SimpleNamespace())
    monkeypatch.setattr(module.trimesh, "load", lambda *args, **kwargs: mesh)
    normalized_center = torch.tensor([[.2, -.1, .05]])
    normalized_scale = torch.tensor([[.01, .02, .003]])
    captured = {}

    def converter(**kwargs):
        model = SimpleNamespace(_xyz=normalized_center.clone(), _scaling=normalized_scale.log())

        def save(_):
            captured["xyz"] = model._xyz.clone()
            captured["scale"] = model._scaling.exp()
        model.save_ply = save
        return model

    monkeypatch.setattr(module, converter_name, converter)
    argv = [module_name, "in_memory_mesh", str(tmp_path / "model.ply")]
    if preserve:
        argv.append("--preserve_original_coords")
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    radius = np.linalg.norm(vertices - vertices.mean(0), axis=1).max()
    expected_xyz = normalized_center if preserve else normalized_center * radius + torch.tensor(vertices.mean(0), dtype=torch.float32)
    torch.testing.assert_close(captured["xyz"], expected_xyz)
    torch.testing.assert_close(captured["scale"], normalized_scale * (1 if preserve else radius))
