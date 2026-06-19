"""
Generate text from the Mini GPT-2 BPE checkpoint.

Usage:
    python generate_bpe_checkpoint.py "Language models" 120
"""

import sys
from pathlib import Path

import torch
import tiktoken

import day3_gpt2_bpe_mini as train_script


CHECKPOINT_PATH = Path(__file__).with_name("checkpoints") / "mini_gpt2_bpe.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["config"]
    train_script.BLOCK_SIZE = cfg["block_size"]
    train_script.BATCH_SIZE = cfg["batch_size"]
    train_script.N_EMBD = cfg["n_embd"]
    train_script.N_HEAD = cfg["n_head"]
    train_script.N_LAYER = cfg["n_layer"]
    train_script.DROPOUT = cfg["dropout"]
    train_script.VOCAB_SIZE = cfg["vocab_size"]
    model = train_script.MiniGPT2BPE().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Language models"
    max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    enc = tiktoken.get_encoding("gpt2")

    model, checkpoint = load_model()
    ids = enc.encode_ordinary(prompt)
    pieces = [enc.decode([i]) for i in ids]

    print(f"device={DEVICE}")
    print(f"checkpoint={CHECKPOINT_PATH}")
    print(f"prompt={prompt!r}")
    print(f"token_ids={ids}")
    print(f"token_pieces={pieces}")
    losses = checkpoint.get("final_losses")
    if losses:
        print(f"saved_train_loss={losses['train']:.4f} saved_val_loss={losses['val']:.4f}")

    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=max_new_tokens)[0].cpu().tolist()

    print()
    print("Generated text:")
    print(enc.decode(generated))


if __name__ == "__main__":
    main()
