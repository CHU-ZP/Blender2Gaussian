# 开发与运行环境

使用 uv 管理 Python、`.venv` 和依赖，固定版本见 `.python-version`、`pyproject.toml`、`uv.lock`。当前方案说明与参数见 [HANDOFF.md](HANDOFF.md)。

## 已验证环境

| 组件 | 配置 |
| --- | --- |
| 系统与 GPU | Linux x86_64，RTX 4060 Laptop（初始化）／A100 40 GiB（联合训练与渲染） |
| Python | 3.13.5 |
| PyTorch | 2.11.0，CUDA 12.8 构建 |
| Blender Python | bpy 5.2.2 |
| Gaussian renderer | gsplat 1.5.3，位于 render 依赖组 |
| 外部编译器 | CUDA Toolkit 12.8、C++ 编译器 |

NumPy/SciPy 用于采样、梯度、邻点与协方差；NVIDIA Warp 用于可见性剔除和相机检查的 GPU 光线求交（无 GPU 时退回 CPU）；Pillow/plyfile 用于纹理和 PLY；bpy/OpenEXR 用于 Mesh 几何、材质和浮点参照；Matplotlib 用于诊断图；trimesh/tqdm 支持原 GLB 接口。其余包及精确依赖关系由锁文件管理。本轮代码清理保留原锁定数值环境，未手工改装 `.venv`。

CUDA 驱动和 Toolkit 是外部工具链，uv 不安装或修改它们。`nvidia-smi` 的 CUDA 版本表示驱动能力，不等于 nvcc 版本。

## 安装与同步

```bash
bash scripts/setup_env.sh --group render
```

脚本查找 CUDA 12.8 的 nvcc（含 `/usr/local`、`~/.local`、PATH 与 Conda 环境），设置 CUDA_HOME/PATH，用 `uv sync --locked` 安装依赖并编译 `_C`，随后运行环境检查。默认 `MAX_JOBS=1` 以适应笔记本资源；`TORCH_CUDA_ARCH_LIST` 未设置时按本机 GPU 检测（A100 为 8.0，RTX 4060 Laptop 为 8.9）。renderer 首次 JIT 编译 gsplat 时使用相同的查找和架构规则。

可以显式指定工具链：

```bash
CUDA_HOME=/path/to/cuda-12.8 bash scripts/setup_env.sh --group render
```

从 Conda 目录找到的只作为 CUDA 编译工具链使用，项目 Python 包仍属于 uv 的 `.venv`。PyTorch 使用配置的官方 cu128 索引，其他包使用项目配置的 PyPI 镜像；本项目关闭自身构建隔离，扩展与运行时使用同一 PyTorch ABI。

日常命令使用 `uv run --no-sync`，避免无意触发重新安装。修改 `csrc/` 或构建配置后用安装脚本重建；必要时：

```bash
bash scripts/setup_env.sh --group render --reinstall-package mesh-to-gs
```

新增依赖使用 `uv add` 或 `uv add --dev`，同时维护 pyproject.toml 和 uv.lock。

## 当前 Blender 主路径

源场景说明见 [data/README.md](data/README.md)，当前方法见 [README](README.md) 和 [best_pipeline.md](docs/best_pipeline.md)。默认执行表面初始化、strength=0.5 纹理协方差及联合 Mesh／图像优化：72 训练视角、优先采样、1500 步，100/200/300/400 步分裂，法线固定。

```bash
uv run --no-sync python scripts/build_model.py data/sk_00918.blend --output-dir outputs/new_sk_00918
uv run --no-sync python scripts/render_gaussians.py outputs/best/sk_00918/model.ply \
  --cameras configs/sk_00918_cameras_render.json --output-dir outputs/best/sk_00918/renders --geometry
```

输出目录必须新建，已保留模型不会覆盖。初始化及 Mesh 标签临时缓存成功后清理。全量联合训练需要 CUDA／gsplat 反向传播，已在 A100 上验证；初始化 CPU 采样配合 Warp GPU 可见性检查。已知场景使用保存的 72 个训练相机，新场景自动生成 24＋48 个通过几何检查的姿态。

当前模型和多视角结果在 `outputs/best/`。`render_gaussians.py` 使用线性合成、白背景、classic、eps2d=0.05；`render_geometry_reference.py` 生成不透明无光照 Mesh 参照，`compare_renders.py` 评估图像、覆盖和深度；`contact_sheet.py` 生成对比图，`audit_model.py` 验证模型参数及分裂 lineage。

## 兼容入口与检查

原 CUDA/Python GLB 转换器继续保留：

```bash
uv run --no-sync python mesh2gs.py input.glb output/model.ply
uv run --no-sync python mesh2gs_python.py input.glb output/reference.ply --device cpu
```

仅运行独立 Python GLB 参考实现时，可以跳过本项目安装和 native 构建：

```bash
uv sync --locked --no-install-project
uv run --no-sync python mesh2gs_python.py input.glb output/reference.ply --device cpu
```

当前包的 Blender 入口仍通过包初始化加载 `_C`，因此上面跳过 native 的方法只适用于独立 `mesh2gs_python.py`，不是完整 Blender 流程的 CPU-only 安装。

```bash
uv run --no-sync pytest -q
uv run --no-sync python scripts/check_environment.py
uv lock --check --offline
uv pip check
```

环境检查实际运行 Python/CUDA 转换和 PLY 读写，检查表面采样与协方差、GPU 光线求交、Principled 访问、EXR 和绘图。测试集中还有真实 GPU 覆盖、颜色合成、Cycles 首交深度、可见性剔除、边界约束投影、移动中心、交错分裂与优先采样的检查。
