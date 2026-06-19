"""
Week 2 Day 2: finer LLM training diagnostics.

This script is still small enough to run locally, but the logging is closer to
what a real LLM training report should inspect:

1. Optimization: loss, gradient norm, update/parameter ratio.
2. Prediction: token accuracy, probability entropy, confidence.
3. Representation: activation RMS in the residual stream.
4. Attention: attention entropy by layer.
5. Module-level movement: embeddings, attention, MLP, LayerNorm, output head.

The goal is not to memorize the code. The goal is to learn what to measure.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


BLOCK_SIZE = 64
BATCH_SIZE = 32
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.10
LEARNING_RATE = 3e-4
STEPS = 1000
EVAL_EVERY = 250
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


TEXT = """
transformer models predict the next token from a context window.
the residual stream carries information through every layer.
attention moves information between token positions.
the mlp transforms each token representation independently.
training loss measures fit to the batch.
validation loss estimates generalization to unseen text.
gradient norms describe the direction and strength of optimization.
parameter updates reveal how much each module actually changes.
scaling law experiments compare loss across model size, data size, and compute.
""".strip() * 100


CHARS = sorted(set(TEXT))
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for ch, i in STOI.items()}
VOCAB_SIZE = len(CHARS)
DATA = torch.tensor([STOI[ch] for ch in TEXT], dtype=torch.long)
SPLIT = int(len(DATA) * 0.9)
TRAIN_DATA = DATA[:SPLIT]
VAL_DATA = DATA[SPLIT:]


def get_batch(split):
    data = TRAIN_DATA if split == "train" else VAL_DATA
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def rms(x):
    return x.float().pow(2).mean().sqrt().item()


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert EMBED_DIM % NUM_HEADS == 0
        self.num_heads = NUM_HEADS
        self.head_dim = EMBED_DIM // NUM_HEADS
        self.qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(EMBED_DIM, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn_entropy = -(attn * (attn.clamp_min(1e-12).log())).sum(dim=-1).mean()
        out = self.dropout(attn) @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, EMBED_DIM)
        return self.proj(out), attn_entropy


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            nn.GELU(),
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        attn_out, attn_entropy = self.attn(self.ln1(x))
        x = x + attn_out
        attn_resid_rms = rms(x)
        x = x + self.mlp(self.ln2(x))
        mlp_resid_rms = rms(x)
        return x, {
            "attn_entropy": attn_entropy.item(),
            "attn_resid_rms": attn_resid_rms,
            "mlp_resid_rms": mlp_resid_rms,
        }


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        _, seq_len = idx.shape
        pos = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)

        block_stats = []
        for block in self.blocks:
            x, stats = block(x)
            block_stats.append(stats)

        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss, block_stats


def parameter_groups(model):
    groups = {
        "embed": [],
        "attn": [],
        "mlp": [],
        "norm": [],
        "head": [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embedding" in name:
            groups["embed"].append(p)
        elif ".attn." in name:
            groups["attn"].append(p)
        elif ".mlp." in name:
            groups["mlp"].append(p)
        elif "ln" in name:
            groups["norm"].append(p)
        elif "head" in name:
            groups["head"].append(p)
    return groups


@torch.no_grad()
def clone_groups(groups):
    return {name: [p.detach().clone() for p in params] for name, params in groups.items()}


def group_norm(params, attr=None):
    total = 0.0
    for p in params:
        tensor = getattr(p, attr) if attr else p
        if tensor is not None:
            total += tensor.detach().float().pow(2).sum().item()
    return math.sqrt(total)


@torch.no_grad()
def group_delta_norm(params, old_params):
    total = 0.0
    for p, old_p in zip(params, old_params):
        total += (p.detach().float() - old_p.float()).pow(2).sum().item()
    return math.sqrt(total)


@torch.no_grad()
def batch_metrics(model, split):
    model.eval()
    x, y = get_batch(split)
    logits, loss, block_stats = model(x, y)
    probs = F.softmax(logits, dim=-1)
    confidence, pred = probs.max(dim=-1)
    token_acc = (pred == y).float().mean().item()
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean().item()
    model.train()
    return {
        "loss": loss.item(),
        "token_acc": token_acc,
        "entropy": entropy,
        "confidence": confidence.mean().item(),
        "block_stats": block_stats,
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train():
    torch.manual_seed(0)
    model = TinyGPT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    groups = parameter_groups(model)
    previous = clone_groups(groups)

    print(f"device={DEVICE} vocab_size={VOCAB_SIZE} parameters={count_parameters(model):,}")
    print(f"block_size={BLOCK_SIZE} embed_dim={EMBED_DIM} layers={NUM_LAYERS} heads={NUM_HEADS}")

    for step in range(STEPS + 1):
        x, y = get_batch("train")
        _, loss, _ = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        before = clone_groups(groups)
        optimizer.step()

        if step % EVAL_EVERY == 0:
            train_m = batch_metrics(model, "train")
            val_m = batch_metrics(model, "val")
            print()
            print(
                f"step={step:4d} "
                f"train_loss={train_m['loss']:.4f} "
                f"val_loss={val_m['loss']:.4f} "
                f"val_ppl={math.exp(val_m['loss']):.2f} "
                f"val_acc={val_m['token_acc']:.3f} "
                f"entropy={val_m['entropy']:.3f} "
                f"confidence={val_m['confidence']:.3f}"
            )

            attn_entropy = [s["attn_entropy"] for s in val_m["block_stats"]]
            resid_rms = [s["mlp_resid_rms"] for s in val_m["block_stats"]]
            print(
                "layers "
                f"attn_entropy={[round(x, 3) for x in attn_entropy]} "
                f"resid_rms={[round(x, 3) for x in resid_rms]}"
            )

            module_parts = []
            for name, params in groups.items():
                pnorm = group_norm(params)
                gnorm = group_norm(params, "grad")
                step_update = group_delta_norm(params, before[name])
                since_log = group_delta_norm(params, previous[name])
                ratio = since_log / (pnorm + 1e-12)
                module_parts.append(
                    f"{name}:p={pnorm:.2f},g={gnorm:.3f},du={step_update:.5f},rel={ratio:.5f}"
                )
            print("modules " + " | ".join(module_parts))
            previous = clone_groups(groups)


if __name__ == "__main__":
    train()
