# 当前保留方案

更新：2026-11-04。`build_model.py` 默认改用无数量预算的 `coverage` 初始化与导出认证，见 [coverage_sampling.md](docs/coverage_sampling.md)。原 `views72_priority1500` 通过 `--sampling legacy` 保留：表面初始化 → strength=0.5 纹理协方差 → 72 视角优先采样／1500 步／交错分裂的联合优化。原 best 模型保持不变。

完整房间的新方案任务已于 2026-11-04 放入 tmux 会话 `mesh2gs-room-coverage`，输出目录为 `outputs/room_coverage_optimization_20261104/`。用户明确要求完整 `data/sk_00918.blend` 房间；此前六案例的真实资产仅为局部构件，不能替代房间渲染。后台 `run.py` 保存源码快照与哈希，依次执行整个场景覆盖初始化、88 视角初始化渲染、72 视角／1500 步联合优化、88 视角原始优化渲染、最终无上限覆盖修复与认证、88 视角最终渲染和 16 独立测试视角评估。使用 GPU 1、1280×720、生产颜色容差 0.02，无对象过滤或点数预算。原 Mesh／best 未改；训练和测试 Mesh 参照经源哈希与相机核对后复用。当前状态必须读取 [status.json](outputs/room_coverage_optimization_20261104/status.json)，查看 [任务页](outputs/room_coverage_optimization_20261104/gallery.html) 和 [日志](outputs/room_coverage_optimization_20261104/job.log)，未完成前不得报告房间渲染或认证已通过。用 `tmux attach -t mesh2gs-room-coverage` 查看后台进程；不要重复启动同一任务。

该房间任务随后按用户明确要求停止，tmux 会话已关闭，`status.json` 为 `stopped`；未经新的恢复指令不得重启。停止于第 1／3407 个对象／材质组，检查点保存 99,126 次单元处理、3,070 个高斯、48,025 个证书叶片及完整待处理队列。房间初始化、优化与渲染均未完成。对第一组 `G-Group_00 / skp_front_default` 的独立检查点副本进行 6 秒 CPU 性能采样，94.2% 时间在 `texture_bounds`，1.5% 在连续覆盖下界；该组使用 `Color_f0f0f0.jpg`，12 面、6 个共面 chart，总面积约 2.514 m²。诊断在 `diagnostics/first_group_profile.json`。当前 `coverage.py` 新增全图颜色最小／最大值缓存：仅在全图范围足以证明足迹误差满足构造容差时跳过重复扫描，否则仍用精确局部界。后台保存的源码快照未修改，该捷径尚未在房间任务中应用或测量提速。

最新六案例验证已完成：`coverage --optimize` 执行覆盖初始化、联合优化、无上限补缺口与最终认证，原始优化模型保存在 `intermediates/optimized/`。最终覆盖全部通过，但优化画质收益不稳定，默认仍只做覆盖初始化。详细结果与限制见 [coverage_optimization_validation.md](docs/coverage_optimization_validation.md)，渲染见 [新采样＋优化图库](outputs/coverage_optimization_20261104/gallery.html)。

两个最终模型在 `outputs/best/{sk_00918,riliaodian}/model.ply`，各有 88 个多视角 RGB／alpha／depth 输出。浏览 `gallery.html`，或查看 `preview_test.png`、`preview_training.jpg`。原测试 PSNR 27.064／28.014 dB，新测试 26.808／28.407 dB。模型 SHA256、完整指标和逐行审计见 `retained_model.json`、`evaluation/`、`model_validation.json`。

使用说明见 [README](README.md)，原保留方法与文件布局见 [best_pipeline.md](docs/best_pipeline.md)，原训练参数见 [joint_quality.json](configs/joint_quality.json)。训练查询／初始化缓存可由源 Blend 再生成。新增覆盖采样实验写入独立目录。

四组 Mesh 重采样消融已完成并归档，见 [resampling_ablation.md](docs/resampling_ablation.md) 与 [机器可读记录](docs/results/resampling_ablation.json)：A 常规分裂、B Mesh 投影、C 区域重采样、D 在 C 相同位置重初始化颜色／先验与协方差。两场景、两种子共 16 次训练、256 张独立测试渲染；其他组精确重放 A 的实际训练输入和父点选择，C／D 共用出生位置。当时 96 项测试、四组 GPU 短程验证和全量审计通过。

全部独立测试 PSNR 两种子均值：房间 A／B／C／D 为 26.874／26.880／26.890／26.908 dB，家具为 28.127／28.114／28.085／28.606 dB。C 单独改位置未得到一致改善；D 相对 A 家具整体 +0.479 dB、固定细节 +0.634 dB，房间整体 +0.034 dB、文字细节 −1.384 dB。最终用户选择两个场景均使用原方案，兼顾细节、速度与维护成本；实验结果如实记录，结论只覆盖两场景、两种子。

首轮因极细三角形重心坐标截断造成 UV 取样不稳定而作废；改用三维最近点及配套重心坐标后，16 次训练均已重跑。两轮实验目录及全部输入／查询缓存、源码快照、模型、渲染和日志已删除，相关方法、修正历史、耗时、协议哈希与审计摘要已归档。相同种子 20261102 的新 A 相对旧 best 全测试差值为房间 −0.0592 dB、家具 −0.0172 dB；输入缓存哈希一致，但模型未逐字节复现，具体来源未定位。

原 best 优化循环耗时为房间 280.917 秒、家具 294.781 秒。实验 A 还生成所有组的共享计划，平均额外成本 106.250／60.912 秒；B／C／D 读取计划，且运行时 GPU 有并发负载，因此历史日志不能作为公平速度排名。更早的初始化与渲染实验见 [experiment_history.md](docs/experiment_history.md)。

保留实现：`initialize_blend_surface.py`、`apply_texture_covariance.py`、`refine_joint.py`、`build_model.py`；共享模块负责 Mesh/UV 提取、表面采样、协方差、可见性、相机、Mesh 标签与最近点、联合优化和视角优先级。保留 renderer／Mesh reference／比较／逐行审计工具，以及原 GLB 接口。专用重采样／投影／重新初始化实现、实验报告／审计入口和测试已删除；联合训练仅支持常规长轴双子分裂。

`provenance/` 中的原训练报告包含已删除中间目录的历史路径，用于记录来源；它们不构成当前渲染依赖。新的模型构建使用同样的实际 Mesh 查询数（每类 1,048,576）和候选数 64，再按训练协议选池。源资产不可修改，既有最终模型不可覆盖。

本次清理后验证：`uv run --no-sync pytest -q` 共 87 项通过；临时完整构建完成 4 次更新和两轮常规分裂（128→132 个高斯），固定法线与成功后的缓存清理通过。两个 best 的 SHA256、1,448 个文件及各 88 视角资源保持不变。详细清理与验证记录见 [original_only.json](docs/validation/original_only.json)。

随后完成两个原模型的连续轨迹渲染，各 288 帧、24 fps、12 秒、1280×720，保存在各 best 的 `trajectory/`。房间采用 13.089 m 室内平滑路线，注视健身区 `[109,132,1]`；家具采用半径 11.246 m 的闭合环绕。完整源 Mesh 与连续移动段的 25 cm 净空检查通过；房间最终朝向重新审计，位置不变。浏览 `trajectory/gallery.html` 或播放 `trajectory/trajectory.mp4`，逐帧 RGB／alpha、地图和报告均保留。原 best 和原 88 视角未改，`outputs/` 根目录仍仅有 `best/`。

新增 `mesh_to_gs/trajectories.py`、`sample_camera_trajectory.py`、`render_trajectory.py` 及四项几何路径测试；现有 renderer 增加可选 `--no-contact-sheet` 以减少长序列内存。91 项测试通过，视频完整解码、PNG 与页面链接检查通过。方法与命令见 [camera_trajectories.md](docs/camera_trajectories.md)，审计见 [camera_trajectories.json](docs/validation/camera_trajectories.json)。采样源几何仅在内存求值，临时构图探查输出已清理。

轨迹随后延长为 30 秒／24 fps／720 帧，并加入左 GT Mesh、右 Gaussian 的同相机对比，输出在各 best 的 `trajectory_comparison/`。房间路径从 13.089 m 增至 29.725 m；家具两圈环绕共 152.658 m，高度在 3.018–4.518 m 间平滑变化。相机分别通过 1,037／3,601 个稠密位置的 25 cm 净空检查。两侧各为 1280×720，顶部增加 32 像素标签栏，对比视频为 2560×752。

新增入口 `render_trajectory_comparison.py` 负责 GT／Gaussian 渲染、报告哈希和帧序检查、视频合成及同步浏览页。`--preview-only` 用 12 组配对关键帧检查构图；`--compose-only` 复用完整帧；`--gpu` 指定设备。GT 沿用 Cycles 16 spp、无光照不透明 BaseColor，新增 `render_geometry_reference.py --rgb-only` 逐帧删除临时 EXR。方法与重现见 [trajectory_comparison.md](docs/trajectory_comparison.md)，验证见 [trajectory_comparison.json](docs/validation/trajectory_comparison.json)。既有模型、源 Blend、88 视角和 12 秒轨迹保留。

加长版验证完成：五项路径测试通过；2,880 张 RGB 和 1,440 张 alpha PNG、两组帧序及四个视频完整解码通过，最大 Mesh 投影误差约 0.0000101 像素。原 1,448 个非轨迹资源、模型／源 Blend／12 秒视频哈希核对通过，临时配对预览已清理。


2026-11-04 覆盖采样实现与渲染验证完成。新默认构建路径使用 `mesh_to_gs/coverage.py` 与 `initialize_coverage_surface.py`，按独立保留源面连续覆盖和质量要求补点，无点数预算、增长百分比或迭代终止上限。完整 R-tree 足迹查询、chart 支撑归属、连续三角形下界、保守边界带、精确周期双线性颜色界、float64 PLY 导出复核与中断回滚恢复已实现；`certify_coverage.py` 可独立复查组合目标。无法证明隐藏的层保守保留，认证后不再训练。

六个实验包含四个受控案例与两个实际构件，18 个匹配相机视角均达到 100% 完整前景／边界覆盖，连续下界≥0.95、独立探针零失败、误差带之外最大 alpha≤0.042990。薄片从 0.43% 提升到 100%；真实房间面板从 76.43% 提升到 100%，脚轮叉架从 4.78% 提升到 100%。面板初次 0.113245 的带外漏出经至少 4σ 的边界构造安全半径修复至 0，重建后 10,866 个高斯；其余五个已完成模型使用此前 3σ 构造，并通过相同质量验收。纹理修复先验证完整颜色范围，再按未满足方向收缩，避免不必要的微米级密度。

实验颜色容差为 0.08，生产默认仍为 0.02；比较对象是初始化结果，未重建／训练两个完整场景。高重复 UV 的首次房间构件只认证源面 0 的 3.22%，27,513 个高斯的完整检查点在独立中断目录保留，状态 `incomplete`，未计入通过结果。最终测试 108 项通过；两个原源资产与两个 best 模型 SHA256 均匹配原记录。查看 [对比图浏览页](outputs/coverage_validation_20261103/gallery.html)、[验证说明](docs/coverage_sampling_validation.md) 与 [机器可读汇总](docs/validation/coverage_sampling.json)。

随后完成新旧初始化的 12 次联合训练，每次 1500 步、12 个新训练相机、3 个独立测试相机、seed 20261104、640×480。表面 GT 逐项一致；所有保留源面按面积监督。两组均在 100／200／300／400 步分裂全部达到 2e-5 梯度阈值的父点，无每轮比例或数量增长上限。训练与分裂使六个新模型的原覆盖证书全部失效，独立探针也发现实际缺口；训练近邻 loss 不能作为覆盖证明。

新增 `coverage_repair.py` 与 `repair_coverage.py`：完整搜索所属 chart 的全部三角形投影中心，重建协方差／颜色／边界约束、保留较低透明度并限制最高为 0.99，再按缺口补点，重新认证实际 float64 PLY。gsplat 1.5.3 对近 1 透明度会在合成当前贡献之前提前退出；两高斯 GPU 实验确认限制到 0.99 后修复该问题，原诊断与首轮结果保留。认证已分批计算且保留全部 R-tree 候选，与逐单元下界一致。检查点绑定源、初始化、优化和 lineage 哈希，同一修复目录加锁，完成状态可快速复核恢复。

最终 62 个原源面全部保留，1,704,714 个证书单元的下界均≥0.95，2,418 个独立探针零失败；18 个测试视角完整前景／边界覆盖均为 100%，带外最大 alpha 为 0。最终相对新初始化 PSNR：平面 +6.784、带孔平面 +3.752、棋盘纹理 +3.478 dB；薄片 −17.721、房间面板 −10.028、脚轮叉架 −15.104 dB。初始化／原始优化／最终总点数为 59,514／538,607／583,910，修复补点 45,303；最终约为初始化的 9.81 倍。收益包含优化与投影、协方差／颜色限制、透明度上限和补点的共同影响，不能单独归因于优化。

最新验证为 116 项测试通过、GPU 透明度回归通过、`build_model --optimize` 短程完整构建通过（1397→1797 点、认证 complete、原始优化模型保留）。最终模型／目标／来源／渲染哈希、有限数值、原源面保留、源码快照和图库链接审计通过，原有 656 个文件哈希未变。完整指标在 [coverage_optimization.json](docs/validation/coverage_optimization.json)，审计在 [final_audit.json](outputs/coverage_optimization_20261104/final_audit.json)。未重建或训练两个完整场景，旧高 UV 重复压力案例仍为 incomplete；生产颜色容差 0.02 未用本次训练验证。恢复运行段和并发耗时不能作为速度排名。
