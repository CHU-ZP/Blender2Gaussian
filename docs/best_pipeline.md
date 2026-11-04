# 当前保留的联合优化流程

保留变体 `views72_priority1500`，参数见 [joint_quality.json](../configs/joint_quality.json)。原始完整模型迁移后未重新训练，SHA256 保持一致；本次对各 88 个视角重新渲染，并重新计算两组独立测试指标。

## 初始化

从 Blender 求值的世界坐标 Mesh 提取几何、UV 和不透明 Principled BaseColor。基础间距 0.024 m、纹理权重 2；按 chart 进行分层采样及局部 kNN 协方差，边界／共面 UV 接缝限制分别为 2／1 sigma。可见性剔除近距离隐藏重合层。然后应用 strength=0.5、目标轴比上限 2 的纹理方向协方差；该阶段保持中心、颜色、opacity、行数和法向厚度。

## 视角与优先级

原 24 个相机保留，另按自由空间、视线和前景比例约束随机生成 48 个相机，新增种子 2026110301。已知场景的最终 72 姿态保存在 `configs/*_cameras_train.json`。Mesh 参照为 1280×720、Cycles 16 spp 的不透明无光照 BaseColor。

训练前评估每个视角的图像 loss，后续对访问视角更新 `EMA = 0.9*EMA + 0.1*render_loss`。高频分数为 Mesh 参照 sRGB 的四邻域 Laplacian 绝对均值，仅在腐蚀一像素后的前景内计算；分数固定。

loss EMA 与高频分数分别截断至自身 95 分位后归一化。每步有放回抽一个视角，概率为 `0.5/72 + 0.25*normalized_loss + 0.25*normalized_detail`。训练种子 20261102，相机抽样使用 seed+1。

## 联合优化和分裂

1500 次 Adam 更新，优化中心、两个切向尺度、面内旋转、线性 RGB 和 opacity。法线固定，SH0，法向厚度为较短切向尺度的 2%；联合阶段不施加 chart 边界协方差投影。

`L = 0.8*L1 + 0.2*(1-SSIM) + 0.1*support + 0.01*surface_colour + centre + 0.01*UV_prior`。

支持场要求 Mesh 表面覆盖、复核空空间为空；颜色与 Mesh UV 对齐；密集局部平面距离和每步 4096 个精确 Mesh 最近点约束中心。每类 Mesh 查询原始采样数是 1,048,576，候选数 64；训练池每类 16384，batch 每类 2048。此处保留实际缓存生成配置，避免把旧训练报告中的未生效新缓存默认值误用于重建。

第 100/200/300/400 步先优化，再按累计平均屏幕中心梯度分裂，阈值 2e-5，每轮最多初始数量的 2%。父高斯被两个子高斯替换，沿切向长轴偏移 ±0.5 sigma，切向尺度除以 1.6，继承颜色和 opacity。存活行保留 Adam 状态，子行动量清零，重建邻点。实际四轮净增约 8%；401–1500 步固定拓扑继续优化。

## 输出与测量

`outputs/best/<scene>/` 仅含保留模型的资源：

- `model.ply`、`lineage.npy`、`retained_model.json`、`model_validation.json`。
- `training_cameras.json` 和独立文件形式的 `training/` 参照缓存。
- `render_cameras.json`、`renders/`：72 个训练姿态＋16 个独立测试姿态；RGB／alpha PNG、float alpha／depth NPY。
- `gallery.html`、`preview_test.png`、`preview_training.jpg`。
- `evaluation/{original_test,fresh_test}/`：各 8 个相机、Mesh 参照、当前模型的测试渲染、指标与对比图。
- `provenance/`：该模型的原训练报告、初始化／纹理协方差／查询生成记录及原完整逐行审计。历史路径用于来源记录，缓存可从源资产重建。
- `trajectory/`：原模型的连续相机轨迹、288 帧 RGB／alpha、12 秒 MP4、浏览页与几何审计；方法见 [camera_trajectories.md](camera_trajectories.md)，不参与测试指标。
- `trajectory_comparison/`：加长的 720 帧轨迹、同相机 GT Mesh／高斯渲染、30 秒并排 MP4 和同步逐帧浏览；方法见 [trajectory_comparison.md](trajectory_comparison.md)。

当前指标见 [best_models.json](results/best_models.json)。房间／家具原测试结果为 27.064／28.014 dB，新测试为 26.808／28.407 dB；训练姿态不计入测试结果。原训练优化耗时分别为 280.917／294.781 秒（4.68／4.91 分钟）。这是历史运行记录，包含当时硬件负载的影响。

最终两个场景均保留本方案；其他方案的实现和输出已删除。[重采样消融记录](resampling_ablation.md)保留方法、结果和速度对比限制。D 在家具上的平均 PSNR 更高，但同时改动颜色／先验与协方差，增加共享计划计算，且损失房间文字细节。当前选择兼顾细节与成本，单种子 best 和两场景实验不能代表跨场景普遍收益。

清理后的 87 项测试、临时完整训练链路和模型完整性检查通过，详见 [清理验证](validation/original_only.json)。临时验证仅使用小规模模型检查构建、分裂与导出链路；两个 best 未重新训练。
