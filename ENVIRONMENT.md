# Mesh2Gaussian 开发环境

项目使用 uv 管理 `.venv`、Python 版本和依赖锁文件。环境覆盖现有 Python 与 CUDA 转换器，以及纹理控制形状、BSDF 正向初始化 SH 所需的数值计算和 Blender 材质访问。算法方案见 [HANDOFF.md](HANDOFF.md)。

## 版本与依赖

当前配置面向 Linux x86_64 和 RTX 4060 Laptop GPU。Python 固定为本机已有的 `3.13.5`，PyTorch 固定为 `2.11.0+cu128`，`bpy` 固定为 `5.2.2`，与本机 Blender 版本一致。其余包的精确版本由 `uv.lock` 记录。

| 依赖 | 用途 |
| --- | --- |
| PyTorch CUDA | 现有 Gaussian 参数、GPU 张量与 C++/CUDA 扩展 |
| NumPy、SciPy | 表面梯度、协方差分解、数值求积和 `scipy.special.sph_harm_y` |
| trimesh、rtree | mesh 读写、采样数据和射线可见性查询 |
| plyfile | Gaussian PLY 读写 |
| Pillow、OpenCV headless | 纹理图像、梯度和 Radiance HDR 浮点图像 |
| OpenEXR | EXR 环境光和浮点材质通道 |
| bpy | 读取 Blender 节点、Principled BSDF、求值几何和烘焙材质 |
| Matplotlib | 科学绘图和实验结果检查 |
| tqdm | 转换进度 |
| setuptools、wheel、packaging、ninja | 构建并加载仓库的 `_C` 扩展 |
| pytest | 后续实现的开发验证，位于默认启用的 `dev` 依赖组 |

CUDA 驱动和 CUDA Toolkit 属于外部工具链，uv 不安装或修改它们。`nvidia-smi` 显示的是驱动支持的 CUDA 版本，不代表已经安装相同版本的 `nvcc`。当前编译要求 Toolkit 12.8，以匹配 PyTorch 的 CUDA 12.8 构建。

## 创建与同步

在仓库根目录运行：

```bash
bash scripts/setup_env.sh
```

脚本执行以下操作：

1. 使用 `CUDA_HOME`，或从常见安装位置和用户 Conda 环境目录中寻找 CUDA 12.8 编译器。
2. 用 `uv sync --locked` 创建或同步 `.venv`，安装项目及其依赖。
3. 针对 RTX 4060 Laptop 的计算能力 `8.9` 编译扩展，默认限制 `MAX_JOBS=1`，降低笔记本编译内存占用。
4. 运行环境检查，验证真实转换、PLY、材质节点、纹理、HDR、SH 积分及射线查询。

从 Conda 目录复用的是 CUDA 编译器和开发文件，项目的 Python 包仍安装在独立 `.venv` 中。也可以显式选择其他 CUDA 12.8 安装：

```bash
CUDA_HOME=/path/to/cuda-12.8 bash scripts/setup_env.sh
```

其他型号 GPU 可以通过 `TORCH_CUDA_ARCH_LIST` 覆盖默认架构。新增依赖时使用 `uv add 包名`，同步更新并保留 `pyproject.toml` 和 `uv.lock`；开发依赖使用 `uv add --dev 包名`。

PyTorch 只从明确配置的官方 `cu128` 索引取得，其他包通过阿里云 PyPI 镜像下载，以适应本机网络。项目构建关闭自身的构建隔离，让扩展使用 `.venv` 中同一份 PyTorch；C++ ABI 由 PyTorch 的 `BuildExtension` 自动选择。

## 日常运行

```bash
uv run mesh2gs.py input.glb output/model.ply
uv run mesh2gs_python.py input.glb output/reference.ply --device cpu
uv run python scripts/check_environment.py
```

也可以先 `source .venv/bin/activate`，再直接调用 `python`。输出路径使用带目录的形式，避免现有 `save_ply` 对空目录字符串的处理问题。

uv 的构建缓存已追踪 `csrc` 源文件。修改 CUDA 源码后通过安装脚本重建，以便设置正确的外部编译工具链；需要强制重建时运行 `bash scripts/setup_env.sh --reinstall-package mesh-to-gs`。

在 Python 中可以直接访问 Blender 材质，也可以调用本机 Blender：

```bash
uv run python -c 'import bpy; print(bpy.app.version_string)'
blender --background scene.blend --python export_materials.py
```

第二种方式使用 Blender 自带的 Python。需要额外包的 Blender 脚本应明确其依赖来源；`.venv` 的包不会自动进入外部 Blender 的解释器。

## 仅使用 Python 参考实现

没有 CUDA Toolkit 时，可以跳过项目扩展的构建，同步锁定的依赖：

```bash
uv sync --locked --no-install-project
uv run --no-sync mesh2gs_python.py input.glb output/reference.ply --device cpu
uv run --no-sync python scripts/check_environment.py --cpu-only
```

此路径仍使用同一份包含 CUDA 运行库的 PyTorch 锁定配置，只跳过 `_C` 构建。`--no-sync` 用于防止 `uv run` 再次尝试构建项目。

## 当前推荐的自适应与联合校准流程

将原始 sk_00918.blend 放到 data 目录，输入说明和哈希见 [data/README.md](data/README.md)。源资产、生成输出、虚拟环境和 native 二进制不纳入 Git；uv.lock、源码、固定相机、质量预设及实验摘要纳入版本管理。

```bash
bash scripts/setup_env.sh --group render
uv run --no-sync python scripts/initialize_blend_surface.py data/sk_00918.blend \
  --output-dir outputs/sk_00918_stage3_directional \
  --calibration-profile configs/basecolor_quality.json
uv run --no-sync python scripts/render_gaussians.py outputs/sk_00918_stage3_directional/model.ply \
  --cameras configs/sk_00918_cameras.json \
  --output-dir outputs/sk_00918_stage3_directional/renders_classic_005 --geometry
```

初始化使用 CPU NumPy/SciPy，渲染使用 RTX 4060 的 gsplat。质量预设按基础间距 0.018 世界单位和纹理/曲率/窄片细节推算数量；当前场景一单位为一米。可用 --base-spacing 调整质量/资源取舍，或 --budget 显式指定数量；--max-budget 默认限制自动数量为 25,000,000。--opacity 覆盖质量档的固定 0.995。

渲染默认 linear 颜色合成、classic 模式与 eps2d=0.05；无需额外参数即可使用已验证设置。为复现旧口径，可显式使用 --color-space display --eps2d 0.3。--rasterize-mode antialiased 仅作为滤波与 opacity 补偿消融选项，当前未选为质量档默认。

PLY 存储 sRGB SH0，渲染前解码到线性空间，合成后编码输出。SuperSplat 可打开 PLY，但其颜色与滤波口径可能不同，精确验收以保存的 gsplat 设置为准。

## Mesh 参照与几何验收

先生成同相机的 Mesh BaseColor/深度参照，再评估。参照脚本支持 --cameras 指向固定相机；具体选项可用 --help 查看。

```bash
uv run --no-sync python scripts/render_geometry_reference.py \
  --cameras configs/sk_00918_cameras.json
uv run --no-sync python scripts/evaluate_surface.py \
  --cameras configs/sk_00918_cameras.json \
  --baseline outputs/sk_00918_stage1_linear/renders \
  --candidate outputs/sk_00918_stage3_directional/renders_classic_005 \
  --output-dir outputs/sk_00918_stage3_directional/evaluation_final \
  --baseline-label 'Before adaptation' --candidate-label 'Retained quality version'
```

评估命令中的 baseline 是本机保存的历史结果；新 checkout 可将 --baseline 指向自己保存的有效对照渲染目录。完整历史汇总脚本 report_material_pipeline.py 同样依赖本机各阶段输出。独立初始化与渲染不依赖这些历史产物。

render_report.json 的 cuda_raster_milliseconds 使用 CUDA events，仅包括光栅化；seconds 还包含结果处理和保存。total_peak_cuda_allocated_bytes 包括加载与渲染期间的 PyTorch 分配峰值，不包括驱动、桌面与其他程序占用。

## 开发验证

```bash
uv run --no-sync python -m pytest -q
uv run --no-sync python scripts/check_environment.py
uv lock --check
uv pip check
uv run --no-sync python scripts/ablate_render_conventions.py
uv run --no-sync python scripts/ablate_sampling_support.py \
  --output-dir outputs/sampling_support_retained
uv run --no-sync python scripts/calibrate_surface.py \
  --output-dir outputs/surface_calibration_retained --objective quality
```

当前校准只搜索基础间距、纹理权重、重叠系数和全局 opacity，共 54 组；不再维护纹理方向拉伸和光学质量反解 opacity。不同倾角/纹理频率参与验证，属于参数选择数据。质量预设与完整场景测量见 [HANDOFF.md](HANDOFF.md) 和 [结构化摘要](docs/validation/sk_00918_basecolor.json)。

此次维护版本的测试为 41 项；移除了仅针对已删除试验功能的三项测试。原始 .blend 的 20,000 点运行检查仅验证当前代码链路，不作为低预算质量结果。完整质量仍引用已验收的 18,924,395 点结果，并做了有效协方差路径及固定 opacity 的等价核验。
