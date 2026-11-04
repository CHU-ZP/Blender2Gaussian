# Mesh2Gaussian

将 Blender Mesh 和 Principled BaseColor 转为表面高斯，支持覆盖认证采样与联合 Mesh／图像优化。原保留训练方案为 **72 视角优先采样、1500 步、允许中心移动、100/200/300/400 步交错分裂、固定法线**（`views72_priority1500`）。

默认构建采用无数量预算的覆盖采样，达到连续表面覆盖与误差要求后导出认证模型。原保留模型使用 Mesh/UV 初始化 → strength=0.5 纹理协方差 → 联合优化，可通过 `--sampling legacy` 复现该流程。原训练参数见 [joint_quality.json](configs/joint_quality.json)，方法见 [best_pipeline.md](docs/best_pipeline.md)。

| 场景 | 最终高斯数 | 原独立测试 PSNR / dB | 新独立测试 PSNR / dB |
| --- | ---: | ---: | ---: |
| 房间 `sk_00918` | 8,190,560 | 27.064 | 26.808 |
| 家具 `riliaodian` | 2,503,124 | 28.014 | 28.407 |

两个场景各保留一个最终模型，均有 **88 个 1280×720 视角渲染**：72 个训练视角和两组各 8 个独立测试视角。训练视角用于展示，测试数值仅来自独立测试视角。

- 房间：[模型](outputs/best/sk_00918/model.ply)、[浏览全部视角](outputs/best/sk_00918/gallery.html)、[测试视角预览](outputs/best/sk_00918/preview_test.png)。
- 家具：[模型](outputs/best/riliaodian/model.ply)、[浏览全部视角](outputs/best/riliaodian/gallery.html)、[测试视角预览](outputs/best/riliaodian/preview_test.png)。

本地浏览器打开 `gallery.html` 可筛选训练／测试视角、查看原尺寸图片和 Mesh 参照。每个场景的 `renders/` 包含 RGB、alpha PNG 及 float alpha/depth NPY；`evaluation/` 保留测试参照和重新核对的指标，`provenance/` 保留该模型的训练记录。模型保持原文件 SHA256。`outputs/` 保留这两个 best 的资源和新的覆盖采样验证实验；更早的中间实验已清理。输入资产和锁定环境仍保留。

两个场景现有 **30 秒、24 fps、720 帧的 GT Mesh／高斯并排轨迹对比**。房间路线长 29.7 m，家具采用两圈变高度环绕；相机通过完整 Mesh 的 25 cm 净空检查。两侧使用相同相机和无光照 BaseColor，各保留 1280×720 原图。浏览 [房间对比](outputs/best/sk_00918/trajectory_comparison/gallery.html)／[家具对比](outputs/best/riliaodian/trajectory_comparison/gallery.html)，或下载 [房间视频](outputs/best/sk_00918/trajectory_comparison/comparison.mp4)／[家具视频](outputs/best/riliaodian/trajectory_comparison/comparison.mp4)。方法和重现命令见 [trajectory_comparison.md](docs/trajectory_comparison.md)。此前的 [12 秒高斯轨迹](docs/camera_trajectories.md)仍保留。

[Mesh 区域重采样四组消融](docs/resampling_ablation.md)和 [早期实验](docs/experiment_history.md)以文档保留。四组消融完成了两场景、两种子的 16 次训练；仅重采样位置未得到一致改善。重采样并重新初始化虽提高家具 PSNR，却损失房间文字细节并增加计划生成成本。最终两个场景均采用原方案。详细方法、逐种子结果、耗时、数值修正历史和审计摘要见 [机器可读记录](docs/results/resampling_ablation.json)，专用实验入口已撤销。

新的 [无数量预算采样方案](docs/coverage_sampling.md)已提供独立入口：按连续覆盖和误差要求补采样，包含边界、孔洞与细薄面；不接受点数预算或增长上限。输出双精度 PLY、独立目标表面、覆盖证书与可恢复检查点。可选 `--optimize` 执行联合优化，再无上限补齐缺口并重新认证；未经修复的优化模型保留在 `intermediates/optimized/`。无法证明隐藏的表面保守保留。

```bash
uv run --no-sync python scripts/build_model.py data/sk_00918.blend --sampling coverage --output-dir outputs/coverage_sk_00918
# 覆盖初始化 → 优化 → 最终覆盖修复与认证
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/build_model.py scene.blend --output-dir outputs/coverage_optimized --optimize --training-cameras cameras_train.json
# 中断后恢复；质量参数必须与初次运行相同
uv run --no-sync python scripts/initialize_coverage_surface.py data/sk_00918.blend --output-dir outputs/coverage_sk_00918 --resume
# 重新检查当前 PLY 的空间覆盖；模型变化会使旧的整体质量证书失效
uv run --no-sync python scripts/certify_coverage.py outputs/coverage_sk_00918
```

匹配相机的 Mesh／旧采样／新采样渲染验证入口为 `scripts/validate_coverage_sampling.py`，报告完整前景、轮廓带、纹理、深度及允许边界带外的漏出。查看 [对比图浏览页](outputs/coverage_validation_20261103/gallery.html)、[验证说明](docs/coverage_sampling_validation.md) 和 [完整指标](docs/validation/coverage_sampling.json)。验证采用四个受控案例和两个真实构件，比较初始化结果，颜色容差为 0.08；生产默认仍为 0.02。本次未重建或训练两个完整场景。

覆盖采样加联合优化的六案例验证也已完成，见 [渲染对比图库](outputs/coverage_optimization_20261104/gallery.html)、[结果与适用范围](docs/coverage_optimization_validation.md) 和 [完整指标](docs/validation/coverage_optimization.json)。最终修复模型的连续覆盖认证及 18 个独立测试视角全部通过；原始优化输出均有空间覆盖缺口。最终画质相对新初始化在平面、带孔平面与棋盘纹理上改善，在薄片与两个真实构件上退化，总点数约为初始化的 9.81 倍。因此默认保持覆盖初始化，优化可选且必须修复认证。这次仍为单种子、颜色容差 0.08 的局部案例验证。

## 安装

Linux x86_64、Python 3.13、PyTorch 2.11.0+cu128、bpy 5.2.2、gsplat 1.5.3、CUDA Toolkit 12.8。全量联合训练及本次渲染在 A100 上验证。

```bash
bash scripts/setup_env.sh --group render
```

参见 [ENVIRONMENT.md](ENVIRONMENT.md) 和 [输入资产](data/README.md)。

## 构建

复现原保留训练流程，为新输出目录执行：

```bash
uv run --no-sync python scripts/build_model.py data/sk_00918.blend --sampling legacy --output-dir outputs/new_sk_00918
```

默认使用对应场景已保存的 72 个训练相机；其他场景自动生成 24＋48 个几何检查通过的相机。`--training-cameras` 可指定相机，`--reference-cache outputs/best/sk_00918/training` 可复用同一场景的参照。`--init-args` 和 `--refine-args` 可传入参数。输出目录已存在时拒绝覆盖。

输出是独立的 `model.ply`、lineage、训练相机／报告和参照图。初始化与 Mesh 查询临时缓存成功后清理；失败时保留诊断输入。Principled 常量或直接图片 BaseColor 支持不透明无光照参照；完整 BSDF、透明材质和高阶 SH 不在此流程内。

## 多视角渲染

```bash
uv run --no-sync python scripts/render_gaussians.py outputs/best/sk_00918/model.ply \
  --cameras configs/sk_00918_cameras_render.json --output-dir outputs/best/sk_00918/renders --geometry
```

家具场景将 `sk_00918` 替换为 `riliaodian`。`configs/*_cameras_test.json` 只含 16 个独立测试姿态，`*_cameras_train.json` 只含 72 个训练姿态。统一采用白背景、线性 RGB 合成再转 sRGB、classic 光栅化、eps2d=0.05。

## 验证

```bash
uv run --no-sync pytest -q
uv run --no-sync python scripts/audit_model.py outputs/best/sk_00918
uv run --no-sync python scripts/audit_model.py outputs/best/riliaodian
uv run --no-sync python scripts/check_environment.py
```

该方案按最终选择保留，兼顾细节与优化成本，覆盖／深度外溢仍有取舍；两个 best 来自单种子、两个已知场景。原 CUDA／Python GLB 转换器 `mesh2gs.py`、`mesh2gs_python.py` 保留。

基于 [hwanhuh/mesh2gaussian](https://github.com/hwanhuh/mesh2gaussian) 和 [torch-splatting](https://github.com/hbb1/torch-splatting)。
