"""Check gsplat against analytic images, then isolate covariance artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData

from render_gaussians import configure_cuda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, default=Path("outputs/sk_00918_validation/model.ply"))
    parser.add_argument("--cameras", type=Path, default=Path("outputs/sk_00918_validation/renders/cameras.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sk_00918_validation/diagnosis"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_cuda()
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization

    torch.set_grad_enabled(False)
    device = "cuda"

    def tensor(value):
        return torch.tensor(value, device=device, dtype=torch.float32)

    def render(means, quats, scales, opacities, colors, view, K, width, height, packed=True, backgrounds=None):
        rgb, alpha, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=view, Ks=K, width=width, height=height,
            near_plane=0.05, far_plane=250, packed=packed, eps2d=0.3,
            radius_clip=0, rasterize_mode="classic", backgrounds=backgrounds,
        )
        return rgb, alpha

    # At the optical axis the perspective Jacobian is diag(f/z, f/z).
    means = tensor([[0, 0, 4]])
    quats = tensor([[1, 0, 0, 0]])
    scales = tensor([[0.1, 0.1, 0.0001]])
    opacities = tensor([0.6])
    colors = tensor([[1, 0, 0]])
    view = torch.eye(4, device=device)[None]
    K = tensor([[[300, 0, 64.5], [0, 300, 64.5], [0, 0, 1]]])
    rgb, alpha = render(means, quats, scales, opacities, colors, view, K, 128, 128)
    measured = alpha[0, :, :, 0].cpu().numpy()
    y, x = np.mgrid[:128, :128]
    variance = (300 * 0.1 / 4) ** 2 + 0.3
    expected = 0.6 * np.exp(-0.5 * ((x - 64) ** 2 + (y - 64) ** 2) / variance)
    expected[expected < 1 / 255] = 0
    error = float(np.max(np.abs(measured - expected)))
    assert error < 5e-5, error
    checks = {"analytic_single_gaussian_max_alpha_error": error}
    rgb_unpacked, alpha_unpacked = render(means, quats, scales, opacities, colors, view, K, 128, 128, packed=False)
    checks["packed_vs_unpacked_max_error"] = float(max((rgb - rgb_unpacked).abs().max(), (alpha - alpha_unpacked).abs().max()))
    assert checks["packed_vs_unpacked_max_error"] < 1e-6
    white, _ = render(means, quats, scales, opacities, colors, view, K, 128, 128,
                      packed=False, backgrounds=tensor([[1, 1, 1]]))
    checks["manual_white_background_max_error"] = float((white - (rgb + 1 - alpha)).abs().max())
    assert checks["manual_white_background_max_error"] < 1e-6

    # A translated/rotated camera sees the same Gaussian in its local frame.
    angle = 0.7
    rotation = tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    c2w = torch.eye(4, device=device)
    c2w[:3, :3], c2w[:3, 3] = rotation, tensor([107, 133, 1.6])
    moved_means = means @ rotation.T + c2w[:3, 3]
    moved_quats = tensor([[np.cos(angle / 2), 0, np.sin(angle / 2), 0]])
    moved, moved_alpha = render(moved_means, moved_quats, scales, opacities, colors, torch.linalg.inv(c2w)[None], K, 128, 128)
    checks["rigid_camera_transform_max_error"] = float(max((moved - rgb).abs().max(), (moved_alpha - alpha).abs().max()))
    assert checks["rigid_camera_transform_max_error"] < 1e-4

    overlap_rgb, overlap_alpha = render(
        tensor([[0, 0, 4], [0, 0, 5]]), quats.repeat(2, 1), scales.repeat(2, 1),
        tensor([0.4, 0.5]), tensor([[1, 0, 0], [0, 1, 0]]), view, K, 128, 128)
    checks["front_to_back_compositing_max_error"] = float(max(
        (overlap_rgb[0, 64, 64] - tensor([0.4, 0.3, 0])).abs().max(),
        (overlap_alpha[0, 64, 64] - 0.7).abs().max()))
    assert checks["front_to_back_compositing_max_error"] < 1e-6
    print("Analytic renderer checks:", json.dumps(checks), flush=True)

    data = PlyData.read(args.ply, mmap="r")["vertex"].data

    def columns(names):
        return torch.from_numpy(np.column_stack([data[name] for name in names]).astype(np.float32)).to(device)

    means = columns(["x", "y", "z"])
    quats = F.normalize(columns([f"rot_{i}" for i in range(4)]), dim=-1)
    scales = columns([f"scale_{i}" for i in range(3)]).exp()
    opacities = torch.from_numpy(np.array(data["opacity"], dtype=np.float32)).to(device).sigmoid()
    colors = (columns([f"f_dc_{i}" for i in range(3)]) * 0.28209479177387814 + 0.5).clamp(0, 1)
    camera_data = json.loads(args.cameras.read_text())
    cameras = [camera_data["cameras"][0], camera_data["cameras"][3]]
    width, height = camera_data["width"], camera_data["height"]
    variants = [
        ("baseline", "Original PLY covariance", quats, scales),
        ("rotation_only", "Conjugated quaternion only", quats * tensor([1, -1, -1, -1]), scales),
        ("cap_3cm", "All sigma capped at 3 cm (diagnostic)", quats, scales.clamp(max=0.03)),
        ("isotropic_1cm", "Isotropic sigma 1 cm (diagnostic)", quats, torch.full_like(scales, 0.01)),
    ]
    thumb_width, thumb_height, label_height = 640, 360, 32
    sheet = Image.new("RGB", (thumb_width * len(cameras), (thumb_height + label_height) * len(variants)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (name, label, variant_quats, variant_scales) in enumerate(variants):
        for col, camera in enumerate(cameras):
            rgb, alpha = render(means, variant_quats, variant_scales, opacities, colors,
                                tensor(camera["view_matrix"])[None], tensor(camera["K"])[None], width, height)
            pixels = ((rgb[0] + 1 - alpha[0]).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            image = Image.fromarray(pixels)
            image.save(args.output_dir / f"{name}_{camera['name']}.png")
            x, y = col * thumb_width, row * (thumb_height + label_height)
            sheet.paste(image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS), (x, y))
            draw.text((x + 8, y + thumb_height + 8), label + " | " + camera["name"], fill="black")
            print(f"Rendered {name}: {camera['name']}", flush=True)
            del rgb, alpha
    sheet.save(args.output_dir / "covariance_ablation.png")
    comparison = Image.new("RGB", (2 * thumb_width, thumb_height + label_height), "white")
    draw = ImageDraw.Draw(comparison)
    for index, (name, label) in enumerate([("baseline", "Original PLY"), ("cap_3cm", "Same centers; sigma <= 3 cm (diagnostic only)")]):
        with Image.open(args.output_dir / f"{name}_{cameras[0]['name']}.png") as source:
            comparison.paste(source.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS), (index * thumb_width, 0))
        draw.text((index * thumb_width + 8, thumb_height + 8), label, fill="black")
    comparison.save(args.output_dir / "scale_comparison.png")
    report = {"source_ply": str(args.ply), "gaussians": len(data), "renderer_checks": checks,
              "controlled_variables": "Same centers, colors, opacities, cameras and renderer; only covariance changes",
              "ablation_is_diagnostic_only": True}
    (args.output_dir / "renderer_checks.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
