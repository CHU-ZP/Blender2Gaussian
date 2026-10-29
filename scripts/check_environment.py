"""Smoke-check conversion, Blender materials, texture tools, HDR and SH numerics."""

import argparse
import importlib.util
import math
from pathlib import Path
import tempfile

import bpy
import cv2
import matplotlib
import numpy as np
import OpenEXR
from plyfile import PlyData
import scipy
from scipy.special import sph_harm_y
import torch
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true", help="Skip the CUDA extension check")
    args = parser.parse_args()

    print(f"Python packages: numpy={np.__version__}, scipy={scipy.__version__}")
    print(f"PyTorch={torch.__version__}, CUDA runtime={torch.version.cuda}")
    print(f"Blender Python={bpy.app.version_string}")
    assert bpy.app.build_options.cycles, "Blender was built without Cycles"

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.eye(3, dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="mesh2gs-check-") as folder:
        folder = Path(folder)
        # Check the actual Python implementation, including PLY serialization.
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("mesh2gs_reference", root / "mesh2gs_python.py")
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        model = reference.mesh_to_gaussmodel_improved(
            vertices, faces, vertex_colors=colors, R=8, k=2,
            device="cpu", max_sh_degree=0, color_mode="none",
        )
        check_model(model, folder / "python.ply")
        print("PASS: Python conversion and PLY round trip (64 Gaussians)")

        if not args.cpu_only:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU unavailable; check the NVIDIA driver and torch installation")
            from mesh_to_gs import mesh_to_gs_cuda

            print(f"GPU={torch.cuda.get_device_name(0)}, capability={torch.cuda.get_device_capability(0)}")
            for mode in ("vertex color", "texture"):
                kwargs = {"vertex_colors": colors} if mode == "vertex color" else {
                    "uvs": vertices[:, :2],
                    "texture": np.full((8, 8, 3), 0.25, dtype=np.float32),
                }
                model = mesh_to_gs_cuda(vertices, faces, R=8, k=2, **kwargs)
                check_model(model, folder / f"cuda-{mode.replace(' ', '-')}.ply")
                if mode == "texture":
                    rgb = model._features_dc.detach().squeeze(-1) * reference.C0 + 0.5
                    torch.testing.assert_close(rgb, torch.full_like(rgb, 0.25))
                print(f"PASS: CUDA conversion and PLY round trip ({mode})")

        material = bpy.data.materials.new("mesh2gs-environment-check")
        try:
            material.use_nodes = True
            bsdf = material.node_tree.nodes.get("Principled BSDF")
            assert bsdf is not None
            for socket in ("Base Color", "Metallic", "Roughness", "Normal", "Alpha"):
                assert socket in bsdf.inputs, socket
        finally:
            bpy.data.materials.remove(material)
        print("PASS: Blender Principled BSDF material access")

        hdr = np.full((8, 8, 3), 4.0, dtype=np.float32)
        hdr_path = folder / "environment.exr"
        OpenEXR.File({}, {"RGB": hdr}).write(str(hdr_path))
        with OpenEXR.File(str(hdr_path)) as image:
            np.testing.assert_allclose(image.channels()["RGB"].pixels, hdr)
        # Radiance HDR is another common environment-map format.
        radiance_path = folder / "environment.hdr"
        assert cv2.imwrite(str(radiance_path), hdr)
        np.testing.assert_allclose(cv2.imread(str(radiance_path), cv2.IMREAD_UNCHANGED), hdr)
        gradient = cv2.Sobel(hdr[:, :, 0], cv2.CV_32F, 1, 0)
        assert np.isfinite(gradient).all()
        print("PASS: floating-point EXR/HDR I/O and texture derivatives")

        # Integrate a known directional function rather than fit rendered images.
        mu, weights = np.polynomial.legendre.leggauss(16)
        theta = np.arccos(mu)
        coefficient = np.sum(weights * (0.5 + 0.2 * mu) * sph_harm_y(1, 0, theta, 0).real) * 2 * np.pi
        np.testing.assert_allclose(coefficient, 0.2 * math.sqrt(4 * np.pi / 3), rtol=1e-12)
        print("PASS: forward SH projection by numerical quadrature")

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        assert mesh.ray.intersects_any([[0.2, 0.2, 1.0]], [[0.0, 0.0, -1.0]])[0]
        print("PASS: mesh visibility ray query with rtree")

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots()
        axis.plot(mu, 0.5 + 0.2 * mu)
        fig.savefig(folder / "sh-check.png")
        plt.close(fig)
        print("PASS: headless scientific plotting")


def check_model(model, path):
    assert model.get_xyz.shape == (64, 3), model.get_xyz.shape
    assert torch.isfinite(model.get_xyz).all()
    assert torch.isfinite(model.get_scaling).all()
    assert torch.isfinite(model._rotation).all()
    torch.testing.assert_close(model.get_xyz[:, 2], torch.zeros_like(model.get_xyz[:, 2]))
    torch.testing.assert_close(model.get_opacity, torch.full_like(model.get_opacity, 0.99))
    model.save_ply(str(path))
    ply = PlyData.read(path)
    assert len(ply["vertex"]) == 64
    for field in ply["vertex"].data.dtype.names:
        assert np.isfinite(ply["vertex"][field]).all(), field


if __name__ == "__main__":
    main()
