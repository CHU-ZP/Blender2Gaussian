# 30 秒 GT Mesh／高斯轨迹对比

两个场景的轨迹由 12 秒延长至 **30 秒、24 fps、720 帧**。逐帧使用完全相同的相机渲染源 Mesh 和保留的 `views72_priority1500` 高斯模型，左侧为 GT Mesh，右侧为 Gaussian。两侧各保留 1280×720 原分辨率，上方增加 32 像素标签栏，最终对比视频为 **2560×752**，H.264、yuv420p、CRF 18。

| 场景 | 路线变化 | 原路程 / m | 新路程 / m | 浏览与视频 |
| --- | --- | ---: | ---: | --- |
| 房间 `sk_00918` | 扩展到吧台、健身区、球台区域 | 13.089 | 29.725 | [浏览页](../outputs/best/sk_00918/trajectory_comparison/gallery.html) · [对比 MP4](../outputs/best/sk_00918/trajectory_comparison/comparison.mp4) |
| 家具 `riliaodian` | 两圈环绕，平滑升高后回到起始高度 | 70.658 | 152.658 | [浏览页](../outputs/best/riliaodian/trajectory_comparison/gallery.html) · [对比 MP4](../outputs/best/riliaodian/trajectory_comparison/comparison.mp4) |

浏览页提供并排视频及共用滑块的逐帧对照，点击任一侧图片可查看原尺寸 PNG。房间的近距离吧台物件、文字和球台，以及家具的不同高度视角都保留在序列中，便于观察原模型的纹理和几何差异。

## 加长轨迹

沿用[轨迹采样方法](camera_trajectories.md)：从完整源 Mesh 构建 BVH，随机种子 `20261103`，连续移动段按最多 5 cm 间距检查，加上半采样间距保护量，确保相机中心距离 Mesh 至少 25 cm。

房间仍在源 bounds 最低点上方 1.5 m 建立 0.2 m 网格，从最大自由空间连通分量的近似直径路线取样。30 秒对应最长 36 m 的路线预算；本次路线平滑后的实际长度为 29.725 m。完整平滑未通过净空检查，采用原路径和滤波路径各占 50% 的组合并再次按弧长采样，通过 1,037 个稠密位置检查。相机保持约 1.489 m 高，全程注视 `[109,132,1]`，起止采用 smoothstep 缓动；最大帧间位移 6.21 cm、旋转 0.7065°。

家具绕源 bbox 中心运行两圈。令 `u = frame / 720`，方位角为 `phase + 4πu`，高度为 `3.018123 + 1.5*(1-cos(2πu))/2` 米：从 3.018 m 平滑升至 4.518 m，再回到起始高度。每个视角均检查 bbox 八个角点入镜，归一化画面范围保留 8% 边距，最终半径为 12.145 m。路径闭合且没有重复保存首帧；包含首尾连接段的 3,601 个稠密位置检查通过。最大帧间位移 21.21 cm、旋转 1.0004°。

## GT 与高斯的对齐

GT 使用源 Blend 的完整 Mesh、原 UV 和 Principled BaseColor。将 BaseColor 接入单位强度 emission，在 Cycles 中以 16 spp 渲染不透明、无光照参照。EXR 中的预乘线性 RGB 合成白背景后转为 sRGB；逐帧保存 RGB PNG 并删除该帧临时 EXR，避免长序列保留大量中间文件。

高斯使用完整原模型、SH0、classic 光栅化和 eps2d=0.05，在线性 RGB 下合成白背景后转为 sRGB。两侧读取同一个 `cameras.json`，共用内外参、裁剪面和帧序号；每个 Mesh 姿态用 Blender 投影矩阵核对 OpenCV 内参。合成视频逐帧左右拼接，不缩放、不插帧。

采样、GT、Gaussian 和对比报告交叉检查源 Blend、模型及相机文件 SHA256，以及帧名、数量和尺寸。轨迹用于展示原模型在连续新视角下的表现，不纳入原有独立测试 PSNR。原模型和源 Blend 未重写，原 88 视角资源及此前 12 秒轨迹保留。

## 输出和重现

新输出位于 `outputs/best/<scene>/trajectory_comparison/`：

- `cameras.json`、`trajectory_report.json`、`path.png`：720 个姿态、几何审计及路线图。
- `mesh/`：720 张 GT RGB PNG 和 `reference_report.json`。
- `renders/`：720 张 Gaussian RGB、720 张 alpha PNG 和 `render_report.json`。
- `comparison.mp4`、`comparison_report.json`、`comparison_preview.jpg`、`comparison_poster.jpg`、`gallery.html`：对比视频和逐帧入口。
- `trajectory.mp4`、`video_report.json`：同路线的单独高斯视频；渲染日志记录实际运行。

采样使用新目录：

```bash
uv run --no-sync python scripts/sample_camera_trajectory.py outputs/best/sk_00918 \
  --output-dir outputs/best/sk_00918/trajectory_comparison_new \
  --frames 720 --fps 24 --target 109 132 1 --gpu 2

uv run --no-sync python scripts/sample_camera_trajectory.py outputs/best/riliaodian \
  --output-dir outputs/best/riliaodian/trajectory_comparison_new \
  --frames 720 --fps 24 --orbit-turns 2 --height-variation 1.5 --gpu 3
```

`--gpu` 可根据空闲 GPU 调整，省略时继承当前 CUDA 设备设置。FFmpeg 可使用系统版本或之前隔离工具环境中的可执行文件：

```bash
trajectory_ffmpeg=$(uv run --no-project --with imageio-ffmpeg==0.6.0 python -c \
  'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')

uv run --no-sync python scripts/render_trajectory_comparison.py \
  outputs/best/sk_00918 outputs/best/sk_00918/trajectory_comparison_new \
  --gpu 2 --ffmpeg "$trajectory_ffmpeg"

uv run --no-sync python scripts/render_trajectory_comparison.py \
  outputs/best/riliaodian outputs/best/riliaodian/trajectory_comparison_new \
  --gpu 3 --ffmpeg "$trajectory_ffmpeg"
```

完整渲染前，可在相同命令末尾添加 `--preview-only`，先生成 12 组配对关键帧；之后去掉该参数生成完整序列。两侧帧和报告都已完成时，`--compose-only` 可仅执行视频合成。已有对比视频会拒绝覆盖。

验证记录见 [trajectory_comparison.json](validation/trajectory_comparison.json)，包括路径检查、相机投影误差、完整视频解码、逐帧 PNG 和原资产哈希检查。

本次五项路径测试通过；两场景共 2,880 张配对 RGB 和 1,440 张 Gaussian alpha PNG 均通过完整性检查。两个并排视频与两个单独高斯视频均完整解码为 720 帧、30 秒、24 fps；Mesh 投影误差最大约 0.0000101 像素。12 组预览和最终对比图均已查看，临时预览已清理。原非轨迹资源仍为 1,448 个文件，原模型、源资产和 12 秒视频哈希一致。
