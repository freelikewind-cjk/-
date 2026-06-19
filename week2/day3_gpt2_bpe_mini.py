"""
Week 2 Day 3: Mini GPT with a GPT-2 BPE tokenizer.

This moves beyond the character-level toy model:

1. Tokenization uses GPT-2's byte-pair encoding through tiktoken.
2. Tokens are word pieces, not single letters.
3. The model is a small GPT-style causal Transformer.
4. The embedding table and output head are weight-tied, like common LMs.

Run on the server:

    ~/llm-training-venv/bin/python day3_gpt2_bpe_mini.py
"""

import argparse
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import tiktoken


BLOCK_SIZE = 128
BATCH_SIZE = 16
N_EMBD = 256
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.10
LEARNING_RATE = 3e-4
STEPS = 800
EVAL_EVERY = 20
EVAL_ITERS = 5
GENERATE_TOKENS = 120
DEFAULT_PROMPT = "Language models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = Path(__file__).with_name("checkpoints") / "mini_gpt2_bpe.pt"


ENC = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = ENC.n_vocab


TEXT = """
Language models learn by predicting the next token from a context.
A tokenizer converts text into integer token IDs.
GPT style models usually use subword tokens instead of single characters.
Attention lets each position read useful previous positions.
The residual stream carries information through the network.
The MLP transforms each token representation after attention.
Training loss measures how surprised the model is by the next token.
Validation loss measures whether the model generalizes beyond the batch.
Checkpoints save model weights, optimizer state, tokenizer name, and config.
During generation, the model predicts one token, appends it, and repeats.
Temperature changes how random the sampling distribution is.
Top k sampling restricts generation to the most likely candidate tokens.
Good debugging means looking at tokens, probabilities, losses, and samples.
""".strip() * 600


DATA = torch.tensor(ENC.encode_ordinary(TEXT), dtype=torch.long)
SPLIT = int(len(DATA) * 0.9)
TRAIN_DATA = DATA[:SPLIT]
VAL_DATA = DATA[SPLIT:]


def get_batch(split):
    data = TRAIN_DATA if split == "train" else VAL_DATA
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        self.n_head = N_HEAD
        self.head_dim = N_EMBD // N_HEAD
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(N_EMBD, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, N_EMBD)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT2BPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(batch_size * seq_len, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
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
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def config():
    return {
        "tokenizer": "gpt2",
        "block_size": BLOCK_SIZE,
        "batch_size": BATCH_SIZE,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "steps": STEPS,
        "vocab_size": VOCAB_SIZE,
    }


def save_checkpoint(model, optimizer, final_losses, checkpoint_path):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config(),
        "final_losses": final_losses,
    }
    with checkpoint_path.open("wb") as f:
        torch.save(checkpoint, f)


def load_checkpoint(model, checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def show_tokenization(prompt):
    ids = ENC.encode_ordinary(prompt)
    pieces = [ENC.decode([i]) for i in ids]
    print(f"prompt={prompt!r}")
    print(f"token_ids={ids}")
    print(f"token_pieces={pieces}")


@torch.no_grad()
def generate_text(model, prompt, max_new_tokens, temperature, top_k):
    idx = torch.tensor([ENC.encode_ordinary(prompt)], dtype=torch.long, device=DEVICE)
    generated = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)[0]
    return ENC.decode(generated.cpu().tolist())


def interactive_generation(model, args):
    print()
    print("Interactive generation")
    print("Type a prompt and press Enter. Type q to quit. Empty prompt uses the default prompt.")
    while True:
        prompt = input("prompt> ").strip()
        if prompt.lower() in {"q", "quit", "exit"}:
            break
        if not prompt:
            prompt = args.prompt
        print()
        print(generate_text(model, prompt, args.generate_tokens, args.temperature, args.top_k))
        print()


def train(args):
    torch.manual_seed(0)
    model = MiniGPT2BPE().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"device={DEVICE}")
    print(f"tokens={len(DATA):,} vocab_size={VOCAB_SIZE:,} parameters={count_parameters(model):,}")
    print(f"block_size={BLOCK_SIZE} n_embd={N_EMBD} layers={N_LAYER} heads={N_HEAD}")
    show_tokenization(args.prompt)

    if args.generate_only:
        load_checkpoint(model, args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
        if args.interactive:
            interactive_generation(model, args)
            return
        print()
        print("Generated text:")
        print(generate_text(model, args.prompt, args.generate_tokens, args.temperature, args.top_k))
        return

    final_losses = None
    for step in range(args.steps + 1):
        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % EVAL_EVERY == 0:
            final_losses = estimate_loss(model)
            print(
                f"step={step:4d} "
                f"train_loss={final_losses['train']:.4f} "
                f"val_loss={final_losses['val']:.4f} "
                f"val_ppl={math.exp(final_losses['val']):.2f}"
            )

        if args.sample_every and step > 0 and step % args.sample_every == 0:
            print()
            print("Sample:")
            print(generate_text(model, args.prompt, args.generate_tokens, args.temperature, args.top_k))
            print()

    if final_losses is None:
        final_losses = estimate_loss(model)
    save_checkpoint(model, optimizer, final_losses, args.checkpoint)
    print(f"Saved checkpoint: {args.checkpoint}")

    print()
    print("Generated text:")
    print(generate_text(model, args.prompt, args.generate_tokens, args.temperature, args.top_k))
    if args.interactive:
        interactive_generation(model, args)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or sample a mini GPT-2 BPE model.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text for tokenization and generation.")
    parser.add_argument("--steps", type=int, default=STEPS, help="Number of training steps.")
    parser.add_argument("--generate-tokens", type=int, default=GENERATE_TOKENS, help="New tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=50, help="Keep only the top k tokens while sampling.")
    parser.add_argument("--sample-every", type=int, default=0, help="Print a sample every N training steps. 0 disables it.")
    parser.add_argument("--generate-only", action="store_true", help="Load checkpoint and generate without training.")
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enter a prompt loop after training or checkpoint load.",
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH, help="Checkpoint path.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
