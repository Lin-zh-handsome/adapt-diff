# RoboTwin 2.0 CNN Diffusion Policy 自适应时间步正式实验

运行位置：`/home/hanjinwei/p1/project/RoboTwin/XPolicyLab/policy/DP/experiments/robotwin2_adaptive_cnn`。本目录保存代码副本；大体积数据、日志、视频与 Checkpoint（检查点）只保留在服务器独立实验目录。

## 数据与模型

直接使用已有的四份 `RoboTwin-<task>-aloha_agilex-joint.zarr`，分别对应 `beat_block_hammer`、`handover_block`、`stack_bowls_three`、`pick_dual_bottles`。每项数据来自 50 条 `demo_clean` 演示，机器人为 Aloha-AgileX，控制方式为 joint（关节控制）。

两组均使用 RoboTwin 官方 XPolicyLab CNN DP：14 维动作、horizon 8、观测步数 3、动作步数 6、100 个 DDPM（去噪扩散概率模型）训练与推理时间步、epsilon prediction（噪声预测）、`down_dims=[256,512,1024]`、batch size 128、600 epoch、seed 42、AdamW 学习率 `1e-4` 和 EMA（指数移动平均）。`use_self_condition=false` 在配置和模型属性中固定。Vanilla 直接调用官方 `compute_loss`；Adaptive 复用 `ActionBetaActor`、Beta 时间步采样、KL 下降反馈、`DeltaQueue`、`SelectKBest` 和 `REINFORCE` 的既有实现。

训练数据仍经过官方 `RobotImageDataset`、专用 batch sampler（批次采样器）和 `dataset.postprocess()`。验证使用固定时间步与固定噪声，覆盖验证集尾批；尾批补足到 128 样本后只计真实样本。两个模式使用同一验证函数。

## 正式运行

服务器 conda 环境：`/home/hanjinwei/p1/envs/robotwin-sc`。在 `/home/hanjinwei/p1/project/RoboTwin/XPolicyLab/policy/DP` 执行：

```bash
CUDA_VISIBLE_DEVICES=0 /home/hanjinwei/p1/envs/robotwin-sc/bin/python -u -m experiments.robotwin2_adaptive_cnn.train --mode vanilla --task beat_block_hammer --run-name robotwin2_adaptdiff_20261008 --device cuda:0
CUDA_VISIBLE_DEVICES=1 /home/hanjinwei/p1/envs/robotwin-sc/bin/python -u -m experiments.robotwin2_adaptive_cnn.train --mode adaptive --task beat_block_hammer --run-name robotwin2_adaptdiff_20261008 --device cuda:0
```

每个任务的两组训练分别占用物理 GPU 0 和 GPU 1，同时运行。后台 `campaign.py` 在两组结束后依次调度其余三个任务；模式间进程、日志、PID、Checkpoint 和异常恢复状态独立。每 25 epoch 保存恢复状态，最终权重为 `runs/<campaign>/<task>/<mode>/checkpoints/600.ckpt`。训练期间只记录训练损失和固定统一验证损失，不启动 RoboTwin 仿真。

四项任务训练结束后，`campaign.py` 用最终 EMA Checkpoint 调用官方 `XPolicyLab/policy/DP/eval.sh`。评测统一使用 `demo_clean`、seen instruction（已见指令）、100 episodes（回合）、seed 42、Aloha-AgileX/joint 和官方 100 步 DDPM 推理。Checkpoint 按官方 `RobotWorkspace` 载荷格式保存；实验目录中的权重经唯一命名的目录链接供官方加载器读取。评测结果实际写入实验目录，代码仓库仅接收小型结果表和曲线 CSV。

## 当前运行

2026-10-08 23:17 左右，`beat_block_hammer` 两组正式训练从头并行启动：Vanilla PID 714359 / GPU 0，Adaptive PID 714360 / GPU 1。两组首轮均写出了非空固定验证损失。调度器 PID 716395，状态日志为 `runs/robotwin2_adaptdiff_20261008/campaign_events.jsonl`。这些 PID 是启动记录；实时状态以服务器进程和日志为准。

原 Can PH 数据与历史日志保留，但已停止 Can 训练，本实验只比较上述四项 RoboTwin 任务。

