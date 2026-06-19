# Week 1: PyTorch Training Basics

## Day 1: MLP learns sin(x)

脚本：`day1_mlp_sin.py`

目标：
1. 运行脚本，看到 loss 下降。
2. 修改 `LEARNING_RATE`、`HIDDEN_SIZE`、`STEPS`。
3. 记录每次实验最后的 loss。

运行：

```powershell
cd D:\大模型训练\llm-training-lab\week1
python day1_mlp_sin.py
```

## Day 2: train/val split and batch training

脚本：`day2_train_val_batch.py`

目标：
1. 理解训练集 `train` 和验证集 `val` 的区别。
2. 理解 `BATCH_SIZE` 是每次喂给模型多少条数据。
3. 同时观察 `train_loss` 和 `val_loss`。

运行：

```powershell
python day2_train_val_batch.py
```

## Day 3: overfitting and weight decay

脚本：`day3_overfit_regularization.py`

目标：
1. 理解过拟合：`train_loss` 低，但 `val_loss` 不一定低。
2. 理解噪声数据会让模型更容易学偏。
3. 尝试 `WEIGHT_DECAY`，观察它对验证 loss 的影响。

运行：

```powershell
python day3_overfit_regularization.py
```
