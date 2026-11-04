"""Smoke-check the supported converters, surface initializer, Blender and EXR."""

import argparse
import importlib.util
from pathlib import Path
import tempfile

import bpy
import matplotlib
import numpy as np
import OpenEXR
from plyfile import PlyData
import scipy
import torch


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

        from mesh_to_gs.surface import charts, sample_component, covariance
        from mesh_to_gs.adaptive import TextureFeatures, prepare_importance
        surface = charts(vertices[faces], faces)
        texture = TextureFeatures(np.full((8, 8, 3), 128, np.uint8))
        np.testing.assert_allclose(prepare_importance(surface, None, .024), 1)
        np.testing.assert_allclose(texture.score(np.array([[.5, .5]]), np.zeros((2, 3)), .024), 0)
        points, normals, _, boundary, spacing, _ = sample_component(surface, 0, 64, 123)
        scales, quaternions, _ = covariance(points, normals, boundary, spacing)
        assert len(points) == 64 and np.isfinite(scales).all() and (scales > 0).all()
        np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-6)
        print("PASS: surface allocation, texture scoring, covariance and rotation")

        from mesh_to_gs.visibility import VisibilityTester
        box = np.array([[[x, y, z] for x, y in corners] for z in (0., 2.) for corners in
                        (((-1, -1), (1, -1), (1, 1)), ((-1, -1), (1, 1), (-1, 1)))])
        tester = VisibilityTester(box, [[-1, -1, 0], [1, 1, 2]], spacing=.5, min_enclosure=0)
        assert len(tester.viewpoints) and abs(tester.normal_gap(np.array([[0., 0, 1]]), np.array([[0., 0, 1]]), 5.)[0] - 1) < 1e-3
        print(f"PASS: GPU ray casting for visibility (Warp, {tester.device})")

        hdr = np.full((8, 8, 3), 4.0, dtype=np.float32)
        hdr_path = folder / "environment.exr"
        OpenEXR.File({}, {"RGB": hdr}).write(str(hdr_path))
        with OpenEXR.File(str(hdr_path)) as image:
            np.testing.assert_allclose(image.channels()["RGB"].pixels, hdr)
        print("PASS: floating-point EXR I/O")

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots()
        axis.scatter(points[:, 0], points[:, 1])
        fig.savefig(folder / "surface-check.png")
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
