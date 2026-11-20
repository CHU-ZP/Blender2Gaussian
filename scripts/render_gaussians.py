"""Render the full Gaussian PLY with gsplat using saved cameras."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from mesh_to_gs.mesh_supervision import digest
from PIL import Image, ImageDraw
from plyfile import PlyData
from mesh_to_gs.color import composite_display_rgb, srgb_to_linear


def cuda_architectures():
    """Compute capabilities of the visible GPUs, e.g. '8.0' (A100) or '8.9' (RTX 4060)."""
    import torch
    if not torch.cuda.is_available():
        return "8.9"
    return ";".join(sorted({"%d.%d" % torch.cuda.get_device_capability(i)
                            for i in range(torch.cuda.device_count())}))


def configure_cuda():
    nvcc_on_path = shutil.which("nvcc")
    roots = [Path(os.environ["CUDA_HOME"])] if os.environ.get("CUDA_HOME") else [
        Path("/usr/local/cuda-12.8"), Path("/usr/local/cuda"),
        Path.home() / ".local/cuda-12.8", Path.home() / ".local/cuda",
        *([Path(nvcc_on_path).resolve().parents[1]] if nvcc_on_path else []),
        *sorted((Path.home() / "miniconda3/envs").glob("*")),
        *sorted((Path.home() / "anaconda3/envs").glob("*")),
    ]
    for root in roots:
        nvcc = root / "bin/nvcc"
        if nvcc.is_file():
            version = subprocess.check_output([str(nvcc), "--version"], text=True)
            if "release 12.8," in version:
                os.environ["CUDA_HOME"] = str(root)
                os.environ["PATH"] = str(root / "bin") + os.pathsep + os.environ["PATH"]
                include = root / "targets/x86_64-linux/include"
                if include.exists():
                    os.environ["CPATH"] = str(include) + os.pathsep + os.environ.get("CPATH", "")
                os.environ.setdefault("MAX_JOBS", "1")
                os.environ.setdefault("TORCH_CUDA_ARCH_LIST", cuda_architectures())
                # Warp or another dependency can import cpp_extension before
                # this function runs. Its cached CUDA_HOME otherwise keeps
                # the earlier toolkit and triggers conflicting gsplat builds.
                import sys
                extension = sys.modules.get('torch.utils.cpp_extension')
                if extension is not None:
                    extension.CUDA_HOME = str(root)
                return
    raise RuntimeError("CUDA Toolkit 12.8 is needed for the initial gsplat build; set CUDA_HOME")


def contact_sheet(images, cameras, path):
    width, height = images[0].size
    thumb_width = 800
    thumb_height = round(height * thumb_width / width)
    label_height = 36
    sheet = Image.new("RGB", (thumb_width * 2, (thumb_height + label_height) * ((len(images)+1)//2)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, camera) in enumerate(zip(images, cameras)):
        x = (index % 2) * thumb_width
        y = (index // 2) * (thumb_height + label_height)
        sheet.paste(image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 10, y + thumb_height + 8), camera["name"] + " | Gaussian splatting", fill="black")
    sheet.save(path)


def centered_view_matrix(matrix, origin):
    """Move camera and centres to the same float64-computed local frame."""
    matrix = np.array(matrix, dtype=np.float64, copy=True)
    matrix[:3, 3] += matrix[:3, :3] @ origin
    return matrix


def validate_appearance_render(report, display_transform, color_space):
    """Reject a second tone map on models trained as final display appearance."""
    if report.get('appearance', {}).get('color_domain') == 'display_linear':
        if display_transform != 'standard' or color_space != 'linear':
            raise ValueError('This model stores display-linear appearance: use '
                             '--display-transform standard --color-space linear; '
                             'exposure/ACES is already represented in its fitted colors')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--geometry", action="store_true", help="Also save float alpha and expected camera-Z depth")
    parser.add_argument('--no-contact-sheet', action='store_true',
                        help='Stream long frame sequences without retaining every image for a contact sheet')
    parser.add_argument("--color-space", choices=("display", "linear"), default="linear",
                        help="Evaluated SH is sRGB; linear decodes before compositing and encodes the final image")
    parser.add_argument('--display-transform', choices=('standard', 'aces_fitted'), default='standard',
                        help='Optional scene-linear display transform, matched to material reference rendering')
    parser.add_argument('--exposure-stops', type=float, default=3.,
                        help='Exposure applied before --display-transform; used only by aces_fitted')
    parser.add_argument('--sh-degree', type=int, choices=range(4),
                        help='Default: infer from standard PLY fields; 0 gives the fitted DC ablation')
    parser.add_argument('--rasterize-mode',choices=('classic','antialiased'),default='classic')
    parser.add_argument('--eps2d',type=float,default=.05,help='Screen covariance filter variance in pixels squared')
    args = parser.parse_args()
    if args.eps2d<=0:parser.error('--eps2d must be positive')
    training_report_path = args.input.resolve().parent/'refinement_report.json'
    training_report = (json.loads(training_report_path.read_text())
                       if training_report_path.exists() else {})
    validate_appearance_render(training_report, args.display_transform, args.color_space)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cameras_path = args.cameras
    camera_data = json.loads(cameras_path.read_text())
    configure_cuda()
    import torch
    import torch.nn.functional as F
    import gsplat
    from gsplat import rasterization
    from gsplat.cuda._wrapper import spherical_harmonics
    from mesh_to_gs.sh_lighting import load_sh, ply_sh_degree

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    # A tiny render triggers JIT and checks the actual CUDA rasterizer first.
    print("Preparing gsplat CUDA rasterizer (initial compilation can take a few minutes)", flush=True)
    warmup, warmup_alpha, _ = rasterization(
        means=torch.tensor([[0.0, 0.0, 3.0]], device=device),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        scales=torch.full((1, 3), 0.15, device=device),
        opacities=torch.ones(1, device=device), colors=torch.tensor([[1.0, 0.0, 0.0]], device=device),
        viewmats=torch.eye(4, device=device)[None],
        Ks=torch.tensor([[[32.0, 0, 16], [0, 32.0, 16], [0, 0, 1]]], device=device), width=32, height=32,
    )
    assert torch.isfinite(warmup).all() and warmup_alpha.max() > 0.1
    del warmup, warmup_alpha
    torch.cuda.reset_peak_memory_stats()
    data = PlyData.read(args.input, mmap="r")["vertex"].data
    count = len(data)
    stored_degree = ply_sh_degree(data.dtype.names)
    degree = stored_degree if args.sh_degree is None else args.sh_degree
    if degree > stored_degree:
        parser.error('Requested SH degree exceeds the PLY coefficient count')
    print(f"Loading all {count:,} Gaussians from {args.input}", flush=True)

    def columns(names):
        return torch.from_numpy(np.column_stack([data[name] for name in names]).astype(np.float32)).to(device)

    world_means = np.column_stack([data[name] for name in 'xyz']).astype(np.float64)
    origin = (world_means.min(0) + world_means.max(0)) / 2
    means = torch.from_numpy((world_means - origin).astype(np.float32)).to(device)
    del world_means
    quats = F.normalize(columns([f"rot_{i}" for i in range(4)]), dim=-1)
    scales = columns([f"scale_{i}" for i in range(3)]).exp()
    opacities = torch.from_numpy(np.array(data["opacity"], dtype=np.float32)).to(device).sigmoid()
    coeffs = None
    if degree:
        coeffs, _ = load_sh(data, device)
    else:
        colors = (columns([f"f_dc_{i}" for i in range(3)]) * 0.28209479177387814 + 0.5).clamp(0, 1)
        if args.color_space == "linear":
            colors = srgb_to_linear(colors)
    loading_peak=torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    images, results = [], []
    width, height = camera_data["width"], camera_data["height"]
    for camera in camera_data["cameras"]:
        print(f"Rendering {camera['name']} at {width}x{height}", flush=True)
        tick = time.perf_counter()
        if coeffs is not None:
            world_camera = np.linalg.inv(np.asarray(camera['view_matrix'], np.float64))[:3, 3]
            camera_position = torch.as_tensor(world_camera-origin, dtype=torch.float32, device=device)
            colors = (spherical_harmonics(degree, means-camera_position, coeffs)+.5).clamp(0, 1)
            if args.color_space == 'linear':
                colors = srgb_to_linear(colors)
        event_start,event_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        event_start.record()
        rgb, alpha, metadata = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=torch.tensor(centered_view_matrix(camera["view_matrix"], origin), device=device, dtype=torch.float32)[None],
            Ks=torch.tensor(camera["K"], device=device, dtype=torch.float32)[None],
            width=width, height=height, near_plane=camera["near"], far_plane=camera["far"],
            packed=True, radius_clip=0, eps2d=args.eps2d, rasterize_mode=args.rasterize_mode,
            render_mode="RGB+ED" if args.geometry else "RGB",
        )
        event_end.record()
        if args.geometry:
            np.save(args.output_dir / (camera["name"] + "_depth.npy"), rgb[0, :, :, 3].cpu().numpy())
            np.save(args.output_dir / (camera["name"] + "_alpha.npy"), alpha[0, :, :, 0].cpu().numpy())
            rgb = rgb[..., :3]
        if args.color_space == 'linear' or args.display_transform == 'aces_fitted':
            rgb = composite_display_rgb(rgb, alpha, display_transform=args.display_transform,
                                        exposure_stops=args.exposure_stops)
        else:
            rgb = rgb + (1 - alpha)
        torch.cuda.synchronize()
        if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
            raise ValueError(f"Non-finite image: {camera['name']}")
        pixels = (rgb[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        image = Image.fromarray(pixels)
        output = args.output_dir / (camera["name"] + ".png")
        image.save(output)
        alpha_pixels = alpha[0, :, :, 0].cpu().numpy()
        Image.fromarray((alpha_pixels.clip(0, 1) * 255).round().astype(np.uint8)).save(args.output_dir / (camera["name"] + "_alpha.png"))
        visible = int(metadata["gaussian_ids"].numel())
        results.append({"camera": camera["name"], "image": str(output), "seconds": time.perf_counter() - tick,
                        "cuda_raster_milliseconds":event_start.elapsed_time(event_end),
                        "projected_gaussians": visible, "mean_alpha": float(alpha_pixels.mean()),
                        "covered_pixels_fraction": float((alpha_pixels > 0.95).mean())})
        if not args.no_contact_sheet:
            images.append(image)
        print(json.dumps(results[-1]), flush=True)
        del rgb, alpha, metadata
    if not args.no_contact_sheet:
        contact_sheet(images, camera_data["cameras"], args.output_dir / "contact_sheet.png")
    report = {"source_ply": str(args.input), "model_sha256": digest(args.input), "gaussians": count, "renderer": "gsplat",
              "renderer_version": gsplat.__version__, "gpu": torch.cuda.get_device_name(0),
              "width": width, "height": height, "sh_degree": degree,
              "stored_sh_degree": stored_degree, "appearance": "sh",
              "rasterize_mode": args.rasterize_mode,
              "eps2d": args.eps2d, "radius_clip": 0, "quaternion_normalization": True,
              "render_origin_world": origin.tolist(),
              "lighting": "Stored sRGB SH color; no additional runtime lighting",
              "stored_color_encoding": "sRGB", "compositing_color_space": args.color_space,
              "trained_color_domain": training_report.get('appearance', {}).get('color_domain'),
              "display_transform": args.display_transform,
              "exposure_stops": args.exposure_stops if args.display_transform == 'aces_fitted' else 0.,
              "depth": "Alpha-normalized expected camera Z (world units)" if args.geometry else None,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "total_peak_cuda_allocated_bytes":max(loading_peak,torch.cuda.max_memory_allocated()),
              "cameras_file": str(cameras_path), "views": results}
    (args.output_dir / "render_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(results)} actual Gaussian renders to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
