# 向 AI 汇报训练异常模板

> 用法：把尖括号里的内容替换成数字、报错或日志即可。没有的数据填 `NA`，不要删字段。

## 1. 一句话问题

- 目标：<例如：训练 124M GPT / 跑 week2 tiny transformer / 2 卡 FSDP 预训练>
- 异常现象：<例如：loss 不降 / NaN / OOM / tokens/s 很低 / 2 卡比单卡慢>
- 我希望你判断：<例如：先查哪个原因、下一步怎么改>

## 2. 实验配置

| 字段 | 数值 |
| --- | --- |
| experiment_id | <EXP-YYYYMMDD-001> |
| 脚本/命令 | `<python train.py ...>` |
| git commit 或代码版本 | `<commit/文件名/改动说明>` |
| 模型规模 | `<参数量 / n_layer / n_head / n_embd>` |
| 数据集 | `<数据名 / token 数 / train-val 划分>` |
| seq_len | `<数字>` |
| batch_size_per_gpu | `<数字>` |
| grad_accum_steps | `<数字>` |
| global_batch_size | `<数字>` |
| precision | `<fp32/bf16/fp16>` |
| num_gpus | `<数字>` |
| optimizer | `<AdamW/SGD/...>` |
| lr | `<数字>` |
| lr_schedule | `<cosine/linear/constant + warmup steps>` |
| weight_decay | `<数字>` |
| grad_clip | `<数字或 NA>` |
| 训练预算 | `<steps / tokens / FLOPs>` |

## 3. 关键指标快照

| 指标 | 起始 | 异常前/当前 | 备注 |
| --- | ---: | ---: | --- |
| train_loss | <数字> | <数字> | <趋势：下降/平台/锯齿/发散> |
| val_loss | <数字或 NA> | <数字或 NA> | <趋势> |
| train-val gap | <数字或 NA> | <数字或 NA> | <是否扩大> |
| grad_norm | <数字> | <数字> | <稳定/尖刺/趋零/爆炸> |
| learning_rate | <数字> | <数字> | <是否符合 schedule> |
| tokens/s | <数字> | <数字> | <是否下降/抖动> |
| MFU | <数字或 NA> | <数字或 NA> | <可选> |
| GPU util | <百分比或 NA> | <百分比或 NA> | <是否忽高忽低> |
| GPU memory | <GB> | <GB> | <是否接近上限/阶梯上升> |

## 4. 最后 50-200 步结构化日志

```csv
step,train_loss,val_loss,lr,grad_norm,tokens_per_s,gpu_mem_gb
<粘贴最后 50-200 行；如果太长，至少贴异常前后各 20 行>
```

## 5. 报错或曲线

完整报错堆栈：

```text
<从第一行 Traceback / RuntimeError 开始粘贴，不要只贴最后一行>
```

loss 曲线/日志链接：

- <wandb/tensorboard/图片路径/NA>

## 6. 我已经试过的修改

| 尝试 | 改动 | 结果 |
| --- | --- | --- |
| 1 | <例如：lr 3e-4 -> 1e-4> | <loss/grad_norm/tokens/s 如何变化> |
| 2 | <例如：开启 grad_clip=1.0> | <结果> |
| 3 | <NA> | <NA> |

## 7. 我怀疑的原因

- <例如：lr 太大 / dataloader 瓶颈 / loss mask 写错 / bf16 数值问题 / checkpoint 没恢复 optimizer>

## 8. 希望你给我的输出

- 请先判断最可能的 2-3 个原因。
- 请给出最小排查顺序。
- 请给出下一次实验只改一处的建议，并说明要记录哪些指标。
