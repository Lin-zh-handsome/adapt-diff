# CVPR 2025 Adaptive Non-uniform Timestep Sampling → Image CNN Diffusion Policy

状态：Step 3 算法实现与合成数据正确性验证完成；未启动完整 Benchmark（基准任务）训练。所有新增代码、测试、结果位于本目录。原始 DiffusionUnetImagePolicy、视觉编码器、ConditionalUnet1D、100 步 DDPM、epsilon-MSE、推理函数未修改。

## 方法与源码对应

| 论文/官方实现 | 本目录实现 | 说明 |
|---|---|---|
| Eq.11；network.py ActorNetwork:74-107；utils/train.py:215-229 | sampler.py ActionBetaActor, sample_timesteps | 以归一化干净动作块 A0 输入三层 Conv1d+pool（原图像三层 Conv2d 的维度适配），softplus 生成 a,b，Beta.sample，经 round(u*(T-1)) 得离散 t；log_prob 仍取连续 u 的 Beta 密度。 |
| Algorithm 1；utils/train.py:296-313 | engine.py process；losses.py epsilon_mse | 主模型仍预测 epsilon，用原 DDPM scheduler.add_noise 和未加权 MSE。Vanilla 模式直接调用原 policy.compute_loss。 |
| Eq.12-14；utils/train.py:275-345 | losses.py feedback_kl；engine.py 前后评估 | 在真正的 optimizer.step 前/后以固定参数状态分别算代表时间步的 KL 反馈；前后重新抽噪，遵从官方实验实现。 |
| Algorithm 2；utils/train.py:237-250；replay_buffer.py | sampler.py DeltaQueue；engine.py _full_values | 单个 A0 扫全部100步，存每步 KL 下降向量及总和，SelectKBest(f_regression) 选3个代表步。 |
| Eq.13；utils/train.py:340-350 | sampler.py reinforce_loss；engine.py | 每40次主模型参数更新执行一次采样器梯度更新；reward 为三个代表步 KL 下降之和加 0.01*Beta 熵，再乘 log_prob，保留官方相对尺度。 |

官方 configs/cifar10.json 采用 Q=5、|S|=3、f_S=40、ent_coef=0.01，本配置取同值。原报告拟用 Q=20 和平均奖励；用户本轮明确要求优先官方实验，因此改用 Q=5 和求和，不把平均后的同一熵系数误当等价算法。现有 DP 使用 cosine 类调度，按论文附录取采样器学习率 1e-3。官方 feature_selector 的尾随逗号会把索引列表变为单元素 tuple；此处返回一维 list[int]，测试检查长度、类型、边界。

## 时间步 0 的连续动作反馈

令 a_bar_t 为累计 alpha，A_t 为带噪动作，A0_hat=(A_t-sqrt(1-a_bar_t)*epsilon_theta)/sqrt(a_bar_t)。对于 t>0，q(A_prev|A_t,A0) 和固定方差 p_theta(A_prev|A_t,O) 的均值分别为 c1*A0+c2*A_t 和 c1*A0_hat+c2*A_t，方差均为 v_t=beta_t*(1-a_bar_prev)/(1-a_bar_t)，故每维 KL=0.5*(mu_q-mu_p)^2/v_t。结果除以 ln(2)，保持官方 bits（比特）单位。

t=0 时 q 的后验方差为零，不能直接套用正态 KL，也不能把离散图像像素的 decoder NLL（解码器负对数似然）搬到连续动作。采用连续动作高斯 p_theta(A0|A_t,O)=Normal(A0_hat,v_1 I)，其中 v_1 是 t=1 的后验方差（对应官方 clipped log variance，裁剪后的对数方差）。每维 NLL_0=0.5*[log(2*pi*v_1)+(A0-A0_hat)^2/v_1]。前后差值中对数项抵消，恰与官方 t=0 KL 代理的参数相关项相同。tests/test_core.py 实测两者的前后差值一致。反馈不改变主模型的 epsilon-MSE。

## 工程约束与使用

- 模式切换仅需 --mode vanilla 或 --mode adaptive。两个模式共用同一原始图像 CNN 策略、数据、runner（评测运行器）、EMA（指数移动平均）及固定均匀时间步验证函数；推理始终是原策略的 predict_action/conditional_sample。
- 梯度累积时，保存属于同一次参数更新的所有微批次；仅在最后一个微批次反传后计算 before，执行一次 optimizer.step，再计算 after 和采样器反馈。f_S 按真实 optimizer.step 次数计；末尾不足一个累积窗口时按实际微批次数修正梯度平均。独立噪声遵从官方，但增大反馈方差。用于反馈的图像编码器以 eval 模式（评估模式）固定中心裁剪，防止随机裁剪使前后数据不同；主训练仍保持原随机裁剪。
- 原 Robomimic 图像数据类若开启 use_cache 会在 HDF5 旁边写缓存。本实验将其关闭；新数据默认放本目录 data/<任务>/ph/image.hdf5，训练输出在 runs/<模式>/<运行名>。
- Checkpoint（检查点）保存 CNN、EMA、优化器、学习率调度器、采样器、采样器优化器、队列及更新计数。恢复是训练状态恢复；随机数和 DataLoader 迭代位置未保存为逐批位级续训状态。

测试命令：在基线工程根目录运行：
```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python -m pytest -q experiments/adaptive_timestep_cnn/tests/test_core.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python -m experiments.adaptive_timestep_cnn.tests.smoke_actual
```

已有结果：8 项单元测试通过；原配置 ResNet18 图像编码器与 ConditionalUnet1D（条件一维 U-Net）在合成图像/动作上完成一次主模型与采样器更新。详见 `runs/unit_test_result.txt` 和 `runs/synthetic_actual_result.json`。这只说明数据流、参数更新和核心公式可执行，不是机器人成功率或训练加速结果。

接入与运行：本仓库只发布实验扩展，不含基线工程。将本目录放到 [Lin-zh-handsome/diff](https://github.com/Lin-zh-handsome/diff) 工程根目录下的 `experiments/adaptive_timestep_cnn`，安装该工程依赖，并准备 Can PH 图像数据 `experiments/adaptive_timestep_cnn/data/can/ph/image.hdf5`。目前没有该图像数据，不能启动正式训练；不会用 `low_dim.hdf5` 代替图像数据。

从基线工程根目录运行：
```bash
python -m experiments.adaptive_timestep_cnn.train --mode vanilla --task can_image --run-name can_seed42
python -m experiments.adaptive_timestep_cnn.train --mode adaptive --task can_image --run-name can_seed42
```
可用 `--device cuda:0` 指定设备，或用 `--resume` 恢复检查点。两种模式应使用各自独立的 GPU 与运行目录。

可行数据来源：Robomimic 官方 Hugging Face 的 v1.5/can/ph/demo_v15.hdf5（https://huggingface.co/datasets/robomimic/robomimic_datasets/tree/main/v1.5/can/ph），再按官方 dataset_states_to_obs.py 生成 84x84、agentview 与 robot0_eye_in_hand 双相机 image.hdf5（https://robomimic.github.io/docs/v0.4/datasets/robosuite.html）；或者先核对第三方现成 image.hdf5 的版本和观测键。数据获取与完整训练属于下一阶段，本阶段没有执行。

未解决：真实图像数据与闭环机器人评测尚未完成；官方前后独立噪声使 KL 下降反馈有抽样方差；连续 Beta 密度经过取整并非严格的离散时间步概率；大模型全时间步探测的实际耗时需在正式训练中统计。
