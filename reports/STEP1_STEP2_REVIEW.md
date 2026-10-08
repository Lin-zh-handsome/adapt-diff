# CNN DP Adaptive Timestep：Step 1–2 审查
日期：2026-10-08；仅审查，未修改代码或启动训练。
工程：/home/hanjinwei/p1/project/diffusion_policy
论文源码：/home/hanjinwei/p1/project/Adaptive-Timestep-Sampler

注：本文记录自适应采样器实现前的阶段 1–2 方案审查；后续原型代码与合成数据验证见 [`experiments/adaptive_timestep_cnn/`](../experiments/adaptive_timestep_cnn/)。

## 现状
Vanilla 目标是 diffusion_policy/policy/diffusion_unet_image_policy.py 的 DiffusionUnetImagePolicy，不是 Transformer/自条件/Flow Matching。配置 diffusion_policy/config/train_diffusion_unet_image_workspace.yaml 使用 ResNet18 编码器、ConditionalUnet1D + FiLM、H=16/To=2/Ta=8、DDPM T=100、squaredcos_cap_v2、fixed_small、epsilon 预测。compute_loss :192-259 归一化动作、torch.randint 均匀采时间步、add_noise、masked MSE。推理 conditional_sample :83-120 保持原样。训练入口 workspace/train_diffusion_unet_image_workspace.py :34-300 负责优化、EMA、runner、日志、Checkpoint。现有验证仍随机采 t，不适合直接比较固定均匀验证损失。p1 中找到 Can PH low_dim.hdf5，但没有 image.hdf5；图像实验尚无数据依据。工作树有既有大量暂存删除，本研究不触碰。

## 论文与官方代码映射
- Eq.11 Beta 采样：ddpm_torch/network.py:74-107，utils/train.py:215-229。CNN 动作块 A0 代替图像 x0，适配输入维度；保留 softplus+1e-5、初始输出层偏置0.5、Beta.sample、round(u*(T-1))、log_prob 和 entropy。
- Eq.12-14 / Algorithm 1：utils/train.py:275-350。主网络仍训练 unweighted epsilon MSE；在参数更新前后计算反馈目标下降，依据连续 Beta 样本 log_prob 以 REINFORCE 更新采样器，每40次主模型 optimizer step 一次。
- Algorithm 2：utils/train.py:237-250、ddpm_torch/replay_buffer.py:3-40。用单个 A0 扫描 T 个时间步构成下降向量；队列 X/Y 用 SelectKBest(f_regression) 选三个代表时间步；启动 S={0,1,2}。
- 附录§9：额外模型前向、全 T 扫描、采样器更新计入实际训练耗时。

## 差异及处理
1. 论文 Eq.14 取 S 的平均；源码 utils/train.py:335 求和。迁移按论文取平均，报告差异。
2. 论文附录 Q=20；configs/cifar10.json 为 Q=5。主配置按论文 Q=20；只在需要解释差异时追加 Q=5 对照。
3. 源码 utils/train.py:250 尾随逗号使选点返回 tuple(list)，与后续时间步循环不兼容；迁移返回一维整数索引。
4. 官方前后反馈重新随机采噪，差值包含采样波动。为估计 Eq.12 的参数更新影响，计划固定同一动作、观测、epsilon 做配对前后计算，并注明这是修正。
5. 官方反馈用图像后验 KL，训练用 MSE；连续机器人动作没有图像离散像素 decoder NLL。按现有 fixed_small 调度器构造连续动作的固定方差 KL / 加权 epsilon MSE 反馈，t=0 单独核对退化后验；不改主网络训练目标。此公式须在编码前核对。
6. 源码 entropy 加到 reward 后乘 log_prob，与独立熵奖励不同。迁移先遵从源码表达式并记录其梯度，不私自换目标。
7. 连续 Beta 密度并非离散时间步 bin 概率；忠实源码使用前者并在结果中说明训练目标偏置。
8. T=100 是本 DP 固有设置，不移植论文 T=1000；f_S 按真实 optimizer step 计。

## 文件级方案
- 新增 diffusion_policy/model/diffusion/adaptive_timestep_sampler.py：动作条件 Beta actor、队列、特征选择、反馈与 REINFORCE（Eq.11-14/Algorithms 1-2）。
- 新增 diffusion_policy/policy/adaptive_diffusion_unet_image_policy.py：扩展 Vanilla 图像策略训练时间步选择；推理复用原策略。
- 最小抽取 diffusion_unet_image_policy.py 的指定 t/epsilon、每样本 loss helper；默认 uniform 行为不变。
- 新增 workspace/train_adaptive_diffusion_unet_image_workspace.py，并为原图像 workspace 抽取 train-step hook；复用数据、EMA、runner、日志、Checkpoint。另存 sampler optimizer、queue 和实际更新计数。
- 新增 config/train_adaptive_diffusion_unet_image_workspace.yaml；沿用 Vanilla CNN/调度/优化器/种子，新增 S=3、f_S=40、Q=20、entropy系数0.01。输出分别放工程根 data/outputs/adaptive_timestep/vanilla 与 adaptive。
- 新增 test/test_adaptive_timestep_sampler.py：核心公式、边界、选点、反馈、两套参数更新及关闭 adaptive 的基线一致性。
- 后续公平实验必须使用同一图像数据、初始化、种子、验证动作/噪声/均匀时间步、推理和 runner 设置；报告迭代数、实际耗时、采样器开销、验证损失和成功率。

本报告不代表代码实现、训练或效果验证完成。
