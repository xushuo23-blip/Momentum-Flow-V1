# 实验一：CelebA Momentum Flow 训练

## part1：初训练 smoke test

目标不是得到高质量图像，而是确认以下链路可以跑通：

- CelebA prompt-image 数据读取
- 冻结 CLIP text encoder 条件输入
- Momentum Flow 两个网络的 loss 和反向传播
- wandb scalar 日志和 preview 图片
- checkpoint 保存到 `Model/checkpoints/exp1_part1_checkpoint/`

运行：

```bash
bash experiments/exp1/part1.sh
```

当前配置只取 `4096` 张图片做小规模跑通（32×32, batch 64, 600 steps）。

如果只想测试训练代码但不上传 wandb：

```bash
python Training/train.py --config configs/exp1/part1_config.py --no-wandb
```

## part2：全量数据 160×160 正式训练

用全部 202k 张 CelebA 图片，160×160 分辨率，128M 参数模型。

| 参数 | 值 |
|------|-----|
| 数据 | 全量 ~202k CelebA，160×160 |
| 模型 | TinyDiT dim=512 depth=12 heads=8 (~64M × 2 = ~128M) |
| batch | 8 |
| 步数 | 25,000 (~1 epoch) |
| 优化 | 双 optimizer (optimizer_r / optimizer_s)，各自独立 backward + clip |

运行：

```bash
bash experiments/exp1/part2.sh
```

checkpoint 保存到 `Model/checkpoints/exp1_part2_160px_checkpoint/`。
