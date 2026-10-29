# Mesh2GaussianSplatting

**Fast Polygonal Mesh to Gaussian Splatting Converter.**
![](./teaser.png)
This project provides a CUDA-accelerated package for converting 3D meshes into Gaussian Splatting representations. 
It also includes a pure Python implementation for reference and comparison.

For ~50K faces mesh: 

| CUDA | Python |
| --- | --- |
| 0.44 sec | 82 sec |

## Features

- Utilizes CUDA for high-performance conversion of meshes to Gaussian splats.
- Includes a pure Python implementation for understanding the conversion logic and for use in environments without a CUDA-compatible GPU.
- Supports meshes with vertex colors or UV-mapped textures.

## Installation

The project environment is managed with [uv](https://docs.astral.sh/uv/). Python and dependency versions are recorded in `.python-version`, `pyproject.toml`, and `uv.lock`.

The current Linux environment targets an RTX 4060 Laptop GPU: Python 3.13, PyTorch 2.11.0 with CUDA 12.8, and Blender Python 5.2.2. Building the extension requires an external CUDA Toolkit 12.8 and a C++ compiler.

```bash
git clone https://github.com/hwanhuh/mesh2gaussian.git
cd mesh2gaussian
bash scripts/setup_env.sh
```

The setup script discovers a compatible CUDA compiler, creates `.venv`, installs the locked dependencies, builds the extension, and runs environment checks. Set `CUDA_HOME` explicitly if your toolkit is elsewhere. See [environment details](ENVIRONMENT.md) for dependency roles, CPU-only setup, and development commands.

## Usage

### Python (CUDA)

The following example demonstrates how to use the CUDA-accelerated converter.

```bash
uv run mesh2gs.py 'mesh.glb' 'output/model.ply'
```

### Pure Python

For comparison or use in non-CUDA environments, you can use the pure Python implementation:

```bash
uv run mesh2gs_python.py 'mesh.glb' 'output/model.ply' --device cpu
```

## How it Works

The conversion process samples points on the surface of the input mesh and represents them as anisotropic 3D Gaussians. 
The density and scale of these Gaussians are determined by the local geometry of the mesh, aiming to create a faithful representation of the original surface.
The CUDA implementation significantly accelerates this process by parallelizing the sampling and Gaussian generation steps on the GPU.

- I use the GaussModel class implemented in [torch-splatting](https://github.com/hbb1/torch-splatting)

## Development handoff

See the [Chinese repository handoff](HANDOFF.md) for the current algorithm, known issues, and proposed improvements using texture-guided Gaussian shapes and forward SH initialization from BSDFs and lighting.

## Blender scene validation

Read the actual Blender scene directly, expand evaluated mesh geometry and face-corner UVs, then run the existing CUDA converter in material batches:

```bash
uv run python scripts/validate_blend.py data/sk_00918.blend
```

This legacy CUDA adapter writes `model.ply`, a point preview, and `report.json` to `outputs/sk_00918_validation/`. It supports constant or directly image-linked Principled Base Color, with fixed opacity 0.99. For the validated adaptive surface path, use the initializer below. See [environment details](ENVIRONMENT.md) for scope and limitations.

Render the resulting Gaussians from four interior viewpoints with the optional gsplat dependency group:

```bash
uv run --group render python scripts/render_gaussians.py
```

The first run compiles gsplat with CUDA Toolkit 12.8. Images, alpha masks, camera parameters, and a render report are saved in `outputs/sk_00918_validation/renders/`. The renderer uses the full PLY and its stored SH0 colors.

## Surface-aware BaseColor initialization

For direct `.blend` conversion with texture/curvature adaptive sampling,
local footprint estimation and joint coverage calibration, use:

```bash
bash scripts/setup_env.sh --group render
uv run --no-sync python scripts/initialize_blend_surface.py data/sk_00918.blend \
  --output-dir outputs/sk_00918_stage3_directional \
  --calibration-profile configs/basecolor_quality.json
uv run --no-sync python scripts/render_gaussians.py outputs/sk_00918_stage3_directional/model.ply \
  --cameras configs/sk_00918_cameras.json \
  --output-dir outputs/sk_00918_stage3_directional/renders_classic_005 --geometry --color-space linear --eps2d 0.05
```

This initializer uses NumPy/SciPy on the CPU and reads `.blend` directly.
The point budget follows world-space target spacing and scene detail, with a
configurable automatic cap of 25 million. Override `--base-spacing` for the
quality/resource tradeoff, or supply `--budget` for an explicit count.
The quality profile uses a 0.018 world-unit base spacing; on this scene one unit
is one meter. Global opacity is 0.995 and can be overridden with `--opacity`.
The renderer defaults to linear color compositing, classic rasterization and
`eps2d=0.05`. The pipeline targets opaque unlit BaseColor; full Principled
appearance and higher-order SH are future work.
The CUDA converter also has corrected random subsequences, normal normalization
and local-to-world wxyz rotations. See [the handoff](HANDOFF.md#当前算法)
and [acceptance commands](ENVIRONMENT.md#当前推荐的自适应与联合校准流程)
for matching unlit Mesh references, float depth/alpha and four-view comparisons.

The maintained path keeps physical longest-edge stratification, world-space
texture/curvature importance, and directional support expansion. Texture-driven
covariance reshaping and density-based per-point opacity were removed after
validation. The input `.blend`, `.venv`, native binaries and generated outputs
stay local; the locked environment, camera configuration and measured result
summary are versioned. See [validation results](docs/validation/sk_00918_basecolor.json)
and [the local asset description](data/README.md).
