# OurMomentumFlow

这是一个用于研究 **text-conditioned Momentum Flow** 的代码工程。当前目标不是训练通用文生图基础模型，而是在一个较窄的目标图像分布上，验证 Momentum Flow 的相空间建模和反向 kinetic 采样是否有效。

当前主线训练两个 TinyDiT 风格的网络：

- `r_theta`：端点方向网络，学习从当前相空间状态指向数据端点的主方向。
- `s_theta_v`：velocity-space score 网络，用于修正反向 kinetic dynamics。

代码默认保留冻结的 CLIP text encoder 作为 prompt 条件入口。即使早期实验只使用固定 prompt 或同类 prompt，也不改成无条件生成，这样后续扩展到更多同类任务时主要改 prompt、数据和 config，不需要重写条件接口。

## 环境依赖

`requirements.txt` 是项目依赖清单，建议上传 GitHub 时保留。

```bash
pip install -r requirements.txt
```

主要依赖包括：

- `torch`
- `torchvision`
- `transformers`
- `ml-collections`
- `wandb`
- `pillow`

## 目录结构

```text
OurMomentumFlow/
  Data/
    train_dataset/      训练图像和 prompt 数据
    sample_dataset/     生成/采样时使用的 prompt 文件
  Training/             训练入口、loss、schedule、state sampling、text encoder、preview sampling
  Sampling/             独立生成模块：text encoder、schedule、reverse ODE/SDE、checkpoint 生成接口
  Model/
    networks/           TinyDiT 等网络结构
    checkpoints/        训练得到的 checkpoint
  configs/              配置文件和配置加载工具
  experiments/          预留实验入口目录，后续按实验编号重新设计
  requirements.txt      Python 依赖
```

## 数据格式

训练数据放在：

```text
Data/train_dataset/<dataset_name>/
```

当前通用格式是 `prompt_image`，也就是一张图像对应一个 prompt。

示例：

```text
Data/train_dataset/my_dataset/
  samples.json
  images/
    sample_00.png
    sample_01.png
```

`samples.json` 格式：

```json
[
  {"image": "images/sample_00.png", "prompt": "generate one target image"},
  {"image": "images/sample_01.png", "prompt": "generate one target image"}
]
```

原始图片可以是混合分辨率。dataloader 会把每张图片 resize 和 center crop 到 `cfg.dataset.image_size`，再归一化到 `[-1, 1]`。同一次实验里需要保持：

- `cfg.dataset.image_size`
- `cfg.model.image_size`
- sampling 阶段的 `image_size`

三者一致。

## 配置文件

所有实验参数都应放在 `configs/` 中。基础配置直接放在 `configs/` 根目录；后续具体实验可以单独开子目录。

已有基础配置：

- `configs/base_training.py`：训练数据路径、image size、模型大小、optimizer、kinetic 超参数、checkpoint 保存路径、wandb 设置等基础训练参数。
- `configs/base_sampling.py`：checkpoint 路径、训练 config 路径、prompt、采样 steps、`tau`、`eta`、输出路径等基础采样参数。
- `configs/loader.py`：配置加载工具，只负责读取带 `get_config()` 的 Python config 文件。

做新实验时，建议复制一份 base config，改成自己的实验配置，不要把超参数硬写在脚本里。

## 训练

可以直接调用核心训练入口：

```bash
python Training/train.py --config configs/base_training.py
```

训练流程大致是：

1. 从 `Data/train_dataset/` 读取图像和 prompt。
2. 用冻结的 CLIP/text encoder 编码 prompt。
3. 构造 Momentum Flow 的相空间训练状态 `y_t=(x_t,v_t)`。
4. 在同一批训练状态上计算两个 loss：
   - `loss_r`：端点方向 MSE（只经过 `r_net`）
   - `loss_s`：NCSN 风格加权 velocity score MSE（只经过 `score_net`）
5. 双 optimizer 各自独立 backward + gradient clip + step。
6. 按 config 保存 checkpoint。

两个网络的 loss 是分开反传和更新的。`r_net` 学习端点方向

```text
r = eps - x0
```

对应的训练目标是普通 MSE：

```text
L_r = E[ || r_theta(y_t, t, tau, e_p) - r ||_2^2 ].
```

`score_net` 学习 velocity-space 条件 score label。代码中先计算

```text
sigma_eff^2(t, tau) = Q_t - C_t^2 / S_t
```

并使用条件高斯下的 label：

```text
s_cond,v =
    [ C_t (x_t - mu_x) - S_t (v_t - mu_v) ]
    / [ S_t Q_t - C_t^2 ].
```

当前 `loss_s` 采用 batch 归一化后的 NCSN 风格权重。对第 `i` 个样本：

```text
w_i = sigma_eff,i^2 = Q_i - C_i^2 / S_i

bar_w_i = w_i / ( (1 / B) * sum_{k=1}^B w_k )
```

最终训练用的 score loss 是

```text
L_s^weighted =
    (1 / B) * sum_{i=1}^B
    (1 / CHW) * sum_{j=1}^{CHW}
    bar_w_i *
    [ s_theta_v(y_ti, t_i, tau_i, e_pi)_j - s_cond,v,i,j ]^2.
```

这里加权的作用不是改变 `s_cond,v` 这个 label，而是让不同 `t,tau` 下的 score 误差进入 loss 时处在更接近的尺度上。代码仍会记录 `loss_s_unweighted`，用于观察未加权 score MSE 的对照值。

checkpoint 默认保存在类似路径：

```text
Model/checkpoints/base_train/checkpoint.200.pt
```

正式实验建议把 `cfg.train.output_dir` 改成单独目录，例如：

```text
Model/checkpoints/exp1/checkpoint.200.pt
```

checkpoint 中保存：

- 当前 step
- `r_net` 权重
- `score_net` 权重

## 采样 / 生成

采样需要已经训练好的 checkpoint 和 prompt。

目前 `experiments/` 入口已清空，采样命令行脚本后续随实验一一起重新设计。当前可复用的生成逻辑在 `Sampling/generation.py`，可以由之后的实验脚本调用。

相关模块：

- `Sampling/reverse.py`：reverse kinetic ODE/SDE 积分器。
- `Sampling/schedules.py`：采样阶段独立使用的 kinetic schedule 和 covariance 工具。
- `Sampling/text_encoder.py`：采样阶段独立使用的冻结 CLIP/T5 text encoder。
- `Sampling/generation.py`：可复用的 checkpoint 生成接口。
- `configs/base_sampling.py`：预留采样默认参数，后续实验脚本可读取。

`Sampling/` 不依赖 `Training/`。也就是说，后续只保留 `configs/`、`Data/`、`Model/` 和 `Sampling/` 时，已经训练好的 checkpoint 仍然可以被加载并生成图片。

`Sampling/generation.py` 是生成编排层：加载 checkpoint、重建网络、加载文本编码器、编码 prompt、检查 batch/prompt 参数，然后调用 `Sampling/reverse.py`。真正的反向 ODE/SDE 公式在 `Sampling/reverse.py`。

## experiments 目录怎么用

`experiments/` 是实验入口目录。实验编号用文件夹表示，同一实验里的不同阶段用文件表示：

```text
experiments/
  exp1/
    part1.sh
    part2.sh
```

共享代码继续放在：

- `Training/`
- `Sampling/`
- `Model/`
- `Data/`

实验自己的参数建议使用同样的层级放在 `configs/`：

```text
configs/exp1/part1_config.py
configs/exp1/part2_config.py
```

实验脚本可以很薄，只负责调用 `Training/`、`Sampling/` 等核心模块。只有当实验需要额外日志、额外评估、ablation hook 时，才在实验脚本里加定制逻辑。

当前已有实验一 part1：

```text
experiments/exp1/part1.sh
configs/exp1/part1_config.py
```

实验一 part1 是初训练 smoke test，目标是跑通训练、wandb、preview 和 checkpoint 保存。运行：

```bash
bash experiments/exp1/part1.sh
```

checkpoint 会保存到：

```text
Model/checkpoints/exp1_part1_checkpoint/
```

## 分辨率说明

代码不是在同一个 batch 里直接训练任意尺寸 tensor。它可以读取不同原始分辨率的图片，但进入模型前会统一转换成一个固定方形尺寸。

建议：

- 调试：`32x32` 或 `64x64`
- 正式像素空间实验：`128x128` 或 `256x256`
- 接近 `512x512`：优先考虑 latent-space 训练，或重新设计更适合高分辨率的架构

提高 image size 时，需要同步检查：

- `dataset.image_size`
- `model.image_size`
- `patch_size`
- batch size
- 显存
- 模型容量

## wandb 使用方式

wandb 的个人账号和 API key 不写进代码。配置里只保存公开的运行选项：

- `wandb.enabled`
- `wandb.project`
- `wandb.entity`
- `wandb.run_name`
- `wandb.mode`
- `wandb.tags`
- `wandb.notes`

当前默认会上传到：

```text
entity = MAIR_HUST
project = Momentum-Flow
```

如果 `wandb.mode="online"`，先在当前机器执行：

```bash
wandb login
```

第一次登录时 wandb 会要求输入 API key。登录成功后，key 会保存在当前服务器用户自己的 wandb 配置目录里；之后同一个服务器用户再次运行训练，通常不需要重复输入 API key。也可以在服务器环境变量里设置 `WANDB_API_KEY`，适合一次性任务、集群 job 或不想交互式登录的场景。

如果别人要把 run 传到这个 entity/project，他需要有这个 wandb entity 的访问权限。不要把自己的 API key 发给别人；应该在 wandb 网页端把对方账号加入对应 team/entity，或者让对方改成自己的 `wandb.entity` 和 `wandb.project`。

不想启用 wandb 时，可以把 config 里的 `wandb.enabled` 改成 `False`，或运行：

```bash
python Training/train.py --config configs/base_training.py --no-wandb
```

## Git 注意事项

不要把大型数据集、wandb 日志、生成图片、大 checkpoint 直接提交到 GitHub，除非这个仓库明确就是用来存 artifact 的。

当前 `.gitignore` 已忽略：

- `.DS_Store`
- Python 缓存
- 虚拟环境
- wandb 日志
- `Model/checkpoints/`
- `Sampling/outputs/`
- 本地大数据目录，例如 `Data/train_dataset/celeba_prompt_image/img_align_celeba/`
