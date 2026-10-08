# adapt-diff

自适应选择扩散时间步训练。

## 当前成果

- [阶段 1–2 方法审查](reports/STEP1_STEP2_REVIEW.md)：记录将论文方法迁移到图像条件机器人动作扩散策略的方案与差异。该报告是实现前的审查快照。
- [CNN Diffusion Policy（卷积神经网络扩散策略）适配原型](experiments/adaptive_timestep_cnn/)：包括自适应时间步采样器、训练步骤编排、损失反馈、配置、测试源文件和已有的合成运行记录。

## RoboTwin 2.0 正式实验

- [四任务 CNN DP 自适应时间步实验](experiments/robotwin2_adaptive_cnn/)：基于现有 XPolicyLab 官方 DP，Vanilla 与 Adaptive 均关闭自条件；训练结束后用官方 eval.sh 独立评测。

## 已停止的 Can 实验

- [Robomimic Can PH 图像正式实验运行说明](RUN_CAN_PH.md)：官方数据下载、两组完整训练命令、GPU 分配和产物路径。

## 项目状态

原型代码与合成数据正确性验证已完成。`runs/unit_test_result.txt` 记录 8 项单元测试通过；`runs/synthetic_actual_result.json` 记录合成数据上的一次模型和采样器更新。这些不是机器人基准任务的训练或效果结果。正式图像基准训练的运行状态及后续结果见正式实验运行说明；尚未产生的闭环成功率不作报告。

本目录是对 [Lin-zh-handsome/diff](https://github.com/Lin-zh-handsome/diff) 中 `diffusion_policy` 包的实验扩展；基线工程、数据和检查点未复制到本仓库。正式训练使用服务器上的原始基线工程与 Can PH 图像数据，运行方式见正式实验说明。

## 参考实现

- [Adaptive-Timestep-Sampler（自适应时间步采样器）上游代码](https://github.com/ku-dmlab/Adaptive-Timestep-Sampler)。本仓库的审查报告引用该项目作为论文参考，没有复制其源码。
