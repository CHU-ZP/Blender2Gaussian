# 原模型的连续轨迹渲染

后续已将轨迹延长至 30 秒，并加入同相机 GT Mesh／高斯并排对照，见 [加长版轨迹对比](trajectory_comparison.md)。下文记录此前的 12 秒高斯轨迹。

房间和家具各生成一组连续相机轨迹，并用保留的 `views72_priority1500` 完整模型渲染。每组为 **288 帧、24 fps、12 秒、1280×720**，视频采用 H.264、yuv420p、CRF 18。源 Blend、模型和既有 88 视角资源保持不变；本次相机用于展示，不参与训练或独立测试指标。

| 场景 | 轨迹 | 路程 / m | 相机高度 / m | 最大帧间旋转 | 浏览与视频 |
| --- | --- | ---: | ---: | ---: | --- |
| 房间 `sk_00918` | 室内平滑开放路线 | 13.089 | 1.489 | 0.744° | [浏览页](../outputs/best/sk_00918/trajectory/gallery.html) · [MP4](../outputs/best/sk_00918/trajectory/trajectory.mp4) |
| 家具 `riliaodian` | 围绕全部家具的闭合圆轨迹 | 70.658 | 3.018 | 1.250° | [浏览页](../outputs/best/riliaodian/trajectory/gallery.html) · [MP4](../outputs/best/riliaodian/trajectory/trajectory.mp4) |

浏览页支持视频播放、滑块逐帧查看，并提供六个代表视角及路线图。房间关键帧包含首尾；家具取第 0/48/96/144/192/240 帧，展示六个不同环绕方向。

## 采样方法

从模型的 `provenance/initializer.json` 读取源 Blend 和场景模式，对源场景求值后在内存中建立完整 Mesh BVH。房间包含 4,936,252 个三角形，家具包含 14,254,847 个三角形；没有创建训练缓存或新的模型。随机种子统一为 `20261103`。

房间在源 bounds 最低点上方 1.5 m 建立 0.2 m XY 网格。筛选射线封闭比例至少 75%、脚下地面高度误差不超过 0.2 m，并具有额外几何净空的网格点。3,691 个有效点中，最大四邻接连通分量包含 3,581 点；从固定种子起点执行两次最远点搜索，取得近似直径的最短路径。取该路线中段，按时长限制长度至最多 14.4 m，进行 Gaussian 平滑、弧长重采样及 smoothstep 起止缓动，再检查整段净空；本次完整平滑通过，最终路线长 13.089 m。

默认室内相机沿平滑后的移动方向看，俯角为 8°。本次关键帧检查发现沿路看会在末段面对近距离文字墙，因此使用 `--target 109 132 1`，全程注视健身区内部的固定世界坐标点。相机位置保持逐元素不变，调整后重新对完整 Mesh 审计全部 288 个姿态，最大帧间旋转降至 0.744°。

家具以源 bounds 中心 `[0.190961, -0.273269, 1.028925]` 为注视点，相机高度为最高几何点上方 0.75 m。随机种子确定起始角度，均匀采样 360°，首尾不重复保存。采用 70° 水平视场角，逐步扩大半径，直到每个姿态下源 bbox 八个角点全部在画面内，并保留归一化画面范围的 8% 边距；最终半径为 11.246 m。该构图展示全场景，因此部分方向留白较多。

## 几何与输出检查

两条路线均要求相机中心距完整源 Mesh 至少 **0.25 m**。不只检查渲染帧位置：每段位移及家具闭环连接段按最多 0.05 m 间距细分，以 float64 三角形／边距离检查，并增加半采样间距的保护量。距离函数的 1-Lipschitz 性质使这项检查覆盖帧间连续线段。房间检查 437 个位置，家具检查 1,441 个位置。

每个姿态还检查旋转正交和粗射线前景，记录帧间位移与角度；最终最大帧间位移为房间 0.0684 m、家具 0.2453 m。采样报告保存源资产、模型与最终相机文件 SHA256，渲染入口拒绝使用与审计不一致的模型或相机。

渲染使用现有 `gsplat` 接口和完整高斯：白背景、线性 RGB 合成后转 sRGB、classic 光栅化、eps2d=0.05。长序列通过 `--no-contact-sheet` 逐帧保存，不在内存保留全部图片。原有渲染命令的默认行为保持不变。

每个 `outputs/best/<scene>/trajectory/` 包含：

- `cameras.json`、`trajectory_report.json`：姿态、时间、采样方法与完整几何审计。
- `renders/`：288 张 RGB PNG、288 张 alpha PNG，以及 `render_report.json`。轨迹展示不额外保存 float depth／alpha NPY。
- `trajectory.mp4`、`video_report.json`、`poster.png`、`preview.jpg`、`path.png`、`gallery.html` 和渲染日志。

最终验证包括 91 项测试、室内门洞 CPU 冒烟检查、关键帧视觉检查、全部 RGB／alpha PNG 的完整性，以及 FFmpeg 完整解码确认 288 帧、12 秒、24 fps、720p。原模型与源资产 SHA256、原有各 88 个视角和页面链接也已核对。详见 [机器可读验证记录](validation/camera_trajectories.json)。

视频用于展示原模型在连续新视角中的表现；局部纹理模糊和缺失表面在结果中保留。

## 重现

使用新目录，采样与渲染入口拒绝覆盖既有轨迹资源：

```bash
CUDA_VISIBLE_DEVICES=2 uv run --no-sync python scripts/sample_camera_trajectory.py \
  outputs/best/sk_00918 --output-dir outputs/best/sk_00918/trajectory_new \
  --frames 288 --fps 24 --width 1280 --height 720 --seed 20261103 \
  --clearance 0.25 --target 109 132 1

CUDA_VISIBLE_DEVICES=3 uv run --no-sync python scripts/sample_camera_trajectory.py \
  outputs/best/riliaodian --output-dir outputs/best/riliaodian/trajectory_new \
  --frames 288 --fps 24 --width 1280 --height 720 --seed 20261103 --clearance 0.25
```

渲染需要带 `libx264` 的 FFmpeg。系统已有 FFmpeg 时可省略 `--ffmpeg`；本次使用隔离的工具环境，不修改项目依赖和锁文件：

```bash
trajectory_ffmpeg=$(uv run --no-project --with imageio-ffmpeg==0.6.0 python -c \
  'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')

CUDA_VISIBLE_DEVICES=2 uv run --no-sync python scripts/render_trajectory.py \
  outputs/best/sk_00918 outputs/best/sk_00918/trajectory_new --ffmpeg "$trajectory_ffmpeg"

CUDA_VISIBLE_DEVICES=3 uv run --no-sync python scripts/render_trajectory.py \
  outputs/best/riliaodian outputs/best/riliaodian/trajectory_new --ffmpeg "$trajectory_ffmpeg"
```

GPU 编号可按机器空闲情况调整。源 Mesh 求值和 BVH 构建通常比实际轨迹渲染更耗时。
