# Mesh2Gaussian 当前版本交接

更新日期：2026 年 10 月 29 日。原始上游基线：`45f867e`。

本版本保留已在 `data/sk_00918.blend` 验证有效的改动。采用几何与 BaseColor 驱动的显式初始化，没有训练网络或逐高斯图像反向优化。历史分析、失败试验与正向 SH 方案保留在 [HANDOFF_HISTORY.md](HANDOFF_HISTORY.md)，其中旧接口不代表当前代码。

## 保留的修改

| 修改 | 作用与证据 |
| --- | --- |
| 坐标与尺度同时恢复 | 防止 Gaussian 中心回到世界坐标而椭球尺度仍留在归一化空间 |
| CUDA 独立随机子序列 | 修复面内线程共享随机流产生重复中心的问题 |
| 小面法线归一化、wxyz 旋转 | 保持正交局部坐标系，椭球第三轴对齐表面法线 |
| 连续表面分片 | 按对象、材质、UV 连续性与 20° 硬边隔离，避免借用附近其他薄片的邻点 |
| 实际最长边分层采样 | 同预算历史实验中，轮廓 F1 从 0.736 提升至 0.818 |
| 世界空间纹理、曲率、窄片重要性 | 按面积和细节自动分配数量，降低对三角形数量的直接依赖 |
| 方向性邻域支撑 | 先查 48 个邻点，方向缺约束或半径过大时扩至 256 个，修复行间覆盖误判 |
| 边界与曲率限制 | 限制跨边界扩张，法向厚度保持为较小切向尺度的 2% |
| 线性颜色和屏幕滤波 | 同一最终 PLY 将 classic 的 eps2d 从 0.3 改为 0.05，PSNR 从 22.31 提至 25.05 dB |
| uv 环境与验证工具 | 锁定依赖，提供原始 Blend 读取、同相机 Mesh 参照、完整 Gaussian 渲染与深度评估 |

## 已移除的试验

- 额外纹理结构张量旋转/拉伸协方差：未证明可靠正收益，压力测试还会降低覆盖。
- 固定 48 近邻的独立方向支撑模式：保留完整的方向扩展方案，不再维护缺少扩展的中间试验。
- 基于光学质量反解逐点 opacity：历史最终结果全部饱和到 0.995，没有形成逐点变化，现简化为全局 opacity。
- 低数量试验预设：原试验深度与轮廓更弱，不作为维护配置；历史文件仍在本机 outputs 中。

渲染脚本保留 display/linear、classic/antialiased 和 eps2d 参数，供受控消融；当前默认是经过验证的 linear、classic、0.05。

## 当前算法

1. 通过 bpy 读取求值后的世界坐标三角形、面角 UV 与材质。图片或常量 BaseColor 在 scene-linear 中插值，再编码为 sRGB SH0。
2. 精确重合顶点恢复几何邻接，按流形边、法线角度和 UV 连续性构建表面片；不会按空间距离焊接附近薄片。
3. 用 UV-to-world Jacobian 将 mip 纹理梯度转换为每米的颜色变化；大面最多 128 个面积探针。曲率用法线夹角除以跨公共边的表面距离估计；窄片使用面积/周长的宽度近似。
4. 基础间距 h 决定低细节密度，面积乘以纹理/曲率/窄片重要性，再加边界权重推算总量。预算允许时每片至少一点，精确整数分配；总数可调，不要求同点数比较。
5. 在三角形内递归沿实际最长边划分，按子区域重要性分配样本；最多四分之一片预算沿边界补点。中心始终位于原 Mesh。
6. 用同片内法线与切平面过滤后的邻点，估计八个方向上的 Voronoi 支撑。发现方向缺口时扩展邻域，并放宽过紧的支撑面积与几何轴比上限。由支撑面积和方向矩构造切向协方差。
7. 对查询到的边界限制其方向的 2*sigma，并限制曲率引起的法线转动；输出单位 wxyz、log-scale 和全局 opacity logit。
8. gsplat 执行完整 Gaussian 投影与 alpha 合成。先把存储的 sRGB 解码为线性颜色，最后编码输出；使用 classic、eps2d=0.05。

当前 [质量配置](configs/basecolor_quality.json) 为：基础间距 0.018 m、纹理权重 8、曲率平方评分、窄片支持、重叠系数 0.75、最大法线转动 0.22 rad、方向性支撑，以及全局 opacity=0.995。

`--base-spacing` 或 `--budget` 可覆盖配置中的数量策略，`--opacity` 可覆盖全局 opacity。自动数量默认上限 2,500 万，并非所有场景都已验证该上限。边界尾部和屏幕滤波仍非紧支撑，不能保证边界外严格为零。

## 入口与复现

- 初始化：`scripts/initialize_blend_surface.py`
- 表面与协方差：`mesh_to_gs/surface.py`
- 材质重要性：`mesh_to_gs/adaptive.py`
- 完整 Gaussian 渲染：`scripts/render_gaussians.py`
- Mesh 深度参照：`scripts/render_geometry_reference.py`
- 指标与图像评估：`scripts/evaluate_surface.py`
- 合成补丁校准：`scripts/calibrate_surface.py`

完整命令见 [ENVIRONMENT.md](ENVIRONMENT.md)。固定相机在 [configs/sk_00918_cameras.json](configs/sk_00918_cameras.json)。原始 Blend 约 402 MB，不纳入 Git；输入哈希见 [data/README.md](data/README.md)。

## 已验证结果

四视角、1280×720、不透明 BaseColor；下表统一 linear、classic、eps2d=0.05。结构化结果保存在 [docs/validation/sk_00918_basecolor.json](docs/validation/sk_00918_basecolor.json)。

| 指标 | 仅校准渲染 | 自适应采样 | 最终质量档 |
| --- | ---: | ---: | ---: |
| Gaussian 数量 | 10,701,421 | 13,556,630 | 18,924,395 |
| PSNR | 24.277 dB | 24.959 dB | 25.045 dB |
| 不透明覆盖率 | 99.738% | 99.683% | 99.806% |
| 深度一致覆盖率 | 96.877% | 97.183% | 96.918% |
| 深度 MAE | 2.359 cm | 2.017 cm | 2.016 cm |
| 深度轮廓 F1 | 0.811 | 0.813 | 0.833 |
| GPU 光栅化时间 | 35.32 ms | 44.41 ms | 58.23 ms |
| PyTorch 显存峰值 | 1.63 GiB | 2.06 GiB | 2.84 GiB |
| CPU 初始化时间 | 201.5 s | 391.0 s | 913.2 s |

较高预算主要换来轮廓、书脊与局部覆盖；从 1,356 万到 1,892 万点的整体 PSNR 增益仅约 0.087 dB，深度一致覆盖还略降。1,356 万点的历史版本性价比更高，其准确源码快照保留在本机 outputs/sk_00918_stage2_adaptive/source 中，当前源码增加了大面纹理积分，重新运行基础自适应命令不会逐点复现那份历史 PLY。

最终历史 PLY 位于本机 outputs/sk_00918_stage3_directional/model.ply，推荐图像为 renders_classic_005，指标为 evaluation_final。此次裁剪保留相同有效参数，固定 opacity 与该 PLY 完全一致；平面、曲面、不均匀行列的协方差、旋转、面积与限幅结果已与裁剪前冻结源码逐项核验一致。当前代码另通过实际 Blend 的 20,000 点运行检查；这只是链路检查，不代替上述完整质量验收。

## 验证口径和限制

- Mesh 使用不透明 BaseColor Emission 排除光照；Gaussian 使用全部 PLY，未替换成点云预览。
- 覆盖检查 alpha>=0.95；深度一致覆盖还要求误差<=max(3 cm,1%参考深度)，排除参考深度轮廓附近 3px。
- Mesh 为首交 camera-Z，Gaussian 为 alpha 归一化的期望中心深度，估计器不同；轮廓 F1 使用深度跳变与 2px 容差。
- CUDA 时间仅包括光栅化，不包括 PLY 载入、PNG/NPY 保存，不等于交互应用帧率；显存为 PyTorch 分配峰值。
- 四个室内视角参与了超参数筛选，尚无独立新场景泛化验证。
- 当前仍有地面文字偏淡、薄杆偏粗等误差。仅支持已提取的直接图片/常量 BaseColor；原场景 13 个材质的 Principled Alpha、完整 BRDF、光照、法线贴图和高阶 SH 未接入。
- 逐点纹理拉伸与逐点 opacity 未保留；后续是否重新引入，应先在覆盖、轮廓与颜色上证明收益。
