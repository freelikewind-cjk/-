# Week 2 Server Runbook

Default training target: use the server, not the local laptop.

Server:

```bash
ssh changjk@166.111.178.7
```

Confirmed hardware:

```text
hostname: dell-PowerEdge-R750xa
GPU: NVIDIA A800-SXM4-80GB
```

Python environment:

```bash
~/llm-training-venv/bin/python
```

Project directory on server:

```bash
~/llm-training-lab/week2
```

Train and save checkpoint:

```bash
cd ~/llm-training-lab/week2
~/llm-training-venv/bin/python day1_tiny_char_transformer.py
```

Load checkpoint and generate:

```bash
cd ~/llm-training-lab/week2
~/llm-training-venv/bin/python generate_from_checkpoint.py "the model " 200
```

Local copy of checkpoint:

```text
D:\大模型训练\llm-training-lab\week2\checkpoints\tiny_char_transformer.pt
```

Working rule:

```text
Use the server for future training runs. Use local execution only for quick code checks or file edits.
```
