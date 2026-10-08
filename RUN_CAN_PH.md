# Robomimic Can PH 图像正式实验

本实验在 `/home/hanjinwei/p1/project/diffusion_policy` 运行；本仓库 `experiments/adaptive_timestep_cnn` 是对应算法代码的发布副本。两组实验仅切换 `--mode`，共用官方完整 Can PH 图像数据和基线配置。

## 数据

官方文件：[Can PH image.hdf5](https://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/image.hdf5)

服务器保存位置：

```text
/home/hanjinwei/p1/project/diffusion_policy/experiments/adaptive_timestep_cnn/data/can/ph/image.hdf5
```

数据使用 `curl --continue-at -` 断点续传；下载完成后以 `h5py` 打开 HDF5（层级数据格式）并确认 `data` 中存在演示轨迹，随后才启动训练。不要把尚未完成的部分文件用于训练。

## 配置与启动

在项目根目录执行：

```bash
cd /home/hanjinwei/p1/project/diffusion_policy
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
  /home/hanjinwei/miniconda3/envs/robodiff/bin/python -u -m experiments.adaptive_timestep_cnn.train \
  --mode vanilla --task can_image --run-name can_seed42 --device cuda:0
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1 \
  /home/hanjinwei/miniconda3/envs/robodiff/bin/python -u -m experiments.adaptive_timestep_cnn.train \
  --mode adaptive --task can_image --run-name can_seed42 --device cuda:0
```

第二个命令通过 `CUDA_VISIBLE_DEVICES=1` 使用物理 GPU 1；进程内部的 `cuda:0` 指向该卡。两张 GPU 与系统内存均足够空闲时可并行；本服务器同时初始化两组评测环境会产生内存压力，因此采用顺序运行。实际任务由实验目录中的 `launch_after_download.sh` 在下载完成后用 `nohup` 启动 Vanilla，`queue_adaptive_after_vanilla.sh` 在 Vanilla 完整结束后自动启动 Adaptive。

两组实验沿用原图像 CNN Diffusion Policy（卷积神经网络扩散策略）配置：seed 42、8000 epoch（完整轮次）、batch size 64、100 个 DDPM（去噪扩散概率模型）时间步、100 步推理、每轮固定均匀时间步验证、每 50 轮闭环评测和 Checkpoint（检查点）。`max_train_steps` 与 `max_val_steps` 均为 null。Adaptive（自适应）仅增加官方方法对应的采样器和反馈更新。

## 产物与指标

```text
experiments/adaptive_timestep_cnn/runs/
  download.log
  formal_launch.log
  vanilla/can_seed42/{launch_config.txt,train.pid,stdout.log,logs.jsonl,latest.ckpt}
  adaptive/can_seed42/{launch_config.txt,train.pid,stdout.log,logs.jsonl,latest.ckpt}
```

`logs.jsonl` 每轮记录 `train_loss`（训练损失）、`uniform_val_loss`（均匀时间步验证损失）、`epoch_seconds`（该轮耗时）、`mean_batch_seconds`（每批平均耗时）、`sampler_updates`（采样器更新次数）和闭环运行器返回的成功率。比较达到相同验证损失或成功率所需的真实墙钟时间，再用批次耗时比较采样器额外开销。只有评测运行器实际产出成功率后才能报告数值。

数据、日志和权重保留在服务器实验目录，不提交到本仓库。


## 2026-10-08 启动记录

- 官方文件完整，HDF5 可读，包含 200 条演示轨迹。
- 按原 `conda_environment.yaml` 将 `robomimic` 固定为 0.2.0、`robosuite` 固定为工程指定的 cheng-chi commit（构建版本 1.2.0），解决评测环境创建错误。
- 实验入口在加载 normalizer（归一化器）后再执行 `model.to(device)` 和 `ema_model.to(device)`，与原训练工作区顺序一致，解决首批数据与模型设备不一致。
- 两次失败启动的日志保留在各模式目录的 `can_seed42_start_failed_20261008_2015`、`can_seed42_start_failed_20261008_2018`。
- 20:30:19 Vanilla 在物理 GPU 0 重新启动，PID 638464；Adaptive 由 `adaptive_queue.log` 记录等待状态，Vanilla 完整结束后自动启动。

这条记录说明启动状态，不代表训练已完成或已有成功率结论。

- 原 Can 图像评测配置 `n_envs=28` 注明需 64 GB 内存。本机约 46 GiB，首次闭环评测中一个环境进程退出并使主进程收到 EOF；正式重启将两组共同的 `n_envs` 调为 8，仅减少并行环境数，保留 6 条训练评测轨迹、50 条测试轨迹、全部原种子、400 步上限和每 50 轮评测频率。该变更不涉及模型、扩散推理或训练步数。失败日志另存于 `can_seed42_start_failed_20261008_2023`。

## 最新状态（2026-10-08 20:34:53，UTC+8）

- Vanilla（基线）与 Adaptive（自适应）运行目录的状态文件均记录在 20:34:53 停止；检查时没有对应训练进程。
- 两份标准输出已完成 46,414 条图像数据的加载进度显示，但逐轮训练日志 `logs.jsonl` 仍为空，未发现检查点（Checkpoint）文件。
- 因此目前没有已完成训练轮次、闭环评测或成功率结果。下载文件、训练日志和数据集继续留在服务器，不随本次说明推送。
