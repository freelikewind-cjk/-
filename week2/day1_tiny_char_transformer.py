"""
Week 2 Day 1: tiny character-level Transformer language model.

This is closer to real LLM training than the sin(x) examples:

1. Text is converted into token IDs.
2. The model sees a context window and predicts the next token.
3. We track train/val loss and generate text from the trained model.

You do not need to memorize this file. Change the knobs first:

BLOCK_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, LEARNING_RATE, STEPS
"""

import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


BLOCK_SIZE = 64
BATCH_SIZE = 32
EMBED_DIM = 96
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.10
LEARNING_RATE = 3e-4
STEPS = 1200
EVAL_EVERY = 200
PARAM_LOG_EVERY = 200
GENERATE_TOKENS = 240
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = Path(__file__).with_name("checkpoints") / "tiny_char_transformer.pt"


TEXT = """
the model studies data and learns a pattern.
the trainer measures loss on training data.
the validator measures loss on new data.
when training loss falls but validation loss rises, the model may overfit.
a language model predicts the next token from previous tokens.
attention lets each token read useful earlier tokens.
larger models can remember more patterns, but they need more data.
good experiments change one knob at a time.
learning rate controls how large each update is.
batch size controls how many examples are used in one update.
weight decay can make the model less eager to memorize noise.
the goal is not to copy the dataset.
the goal is to predict unseen text well.
""".strip() * 80


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


CHARS, STOI, ITOS = build_vocab(TEXT)
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


class TinyCharTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=4 * EMBED_DIM,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.ln = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        positions = torch.arange(seq_len, device=idx.device)

        x = self.token_embedding(idx) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=idx.device) * float("-inf"),
            diagonal=1,
        )
        x = self.blocks(x, mask=mask)
        x = self.ln(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, VOCAB_SIZE),
                targets.reshape(batch_size * seq_len),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -BLOCK_SIZE:]
            logits, _ = self(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(20):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def decode(ids):
    return "".join(ITOS[int(i)] for i in ids)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def model_config():
    return {
        "block_size": BLOCK_SIZE,
        "batch_size": BATCH_SIZE,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "steps": STEPS,
        "vocab_size": VOCAB_SIZE,
    }


def save_checkpoint(model, optimizer, final_losses):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model_config(),
        "chars": CHARS,
        "stoi": STOI,
        "itos": ITOS,
        "text": TEXT,
        "final_losses": final_losses,
    }
    with CHECKPOINT_PATH.open("wb") as f:
        torch.save(checkpoint, f)


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def clone_parameters(model):
    return [p.detach().clone() for p in trainable_parameters(model)]


@torch.no_grad()
def parameter_norm(model):
    total = 0.0
    for p in trainable_parameters(model):
        total += p.detach().float().pow(2).sum().item()
    return math.sqrt(total)


@torch.no_grad()
def parameter_delta_norm(model, snapshot):
    total = 0.0
    for p, old_p in zip(trainable_parameters(model), snapshot):
        total += (p.detach().float() - old_p.float()).pow(2).sum().item()
    return math.sqrt(total)


def gradient_norm(model):
    total = 0.0
    for p in trainable_parameters(model):
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(total)


def train():
    torch.manual_seed(0)

    model = TinyCharTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    last_log_snapshot = clone_parameters(model)

    print(f"device={DEVICE}")
    print(f"vocab_size={VOCAB_SIZE} parameters={count_parameters(model):,}")
    print(f"block_size={BLOCK_SIZE} embed_dim={EMBED_DIM} layers={NUM_LAYERS} heads={NUM_HEADS}")

    final_losses = None

    for step in range(STEPS + 1):
        x, y = get_batch("train")
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        current_grad_norm = gradient_norm(model)

        before_update = None
        if step % PARAM_LOG_EVERY == 0:
            before_update = clone_parameters(model)

        optimizer.step()

        if step % EVAL_EVERY == 0:
            losses = estimate_loss(model)
            final_losses = losses
            ppl = math.exp(losses["val"])
            current_param_norm = parameter_norm(model)
            update_since_log = parameter_delta_norm(model, last_log_snapshot)
            relative_update = update_since_log / (current_param_norm + 1e-12)
            step_update = parameter_delta_norm(model, before_update)
            print(
                f"step={step:4d} "
                f"train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f} "
                f"val_ppl={ppl:.2f} "
                f"param_norm={current_param_norm:.3f} "
                f"grad_norm={current_grad_norm:.3f} "
                f"step_update={step_update:.6f} "
                f"update_since_log={update_since_log:.4f} "
                f"relative_update={relative_update:.6f}"
            )
            last_log_snapshot = clone_parameters(model)

    if final_losses is None:
        final_losses = estimate_loss(model)

    save_checkpoint(model, optimizer, final_losses)
    print()
    print(f"Saved checkpoint: {CHECKPOINT_PATH}")

    prompt = "the model "
    start = torch.tensor([[STOI[ch] for ch in prompt]], dtype=torch.long, device=DEVICE)
    generated = model.generate(start, GENERATE_TOKENS)[0].cpu()
    print()
    print("Generated text:")
    print(decode(generated))


if __name__ == "__main__":
    train()
