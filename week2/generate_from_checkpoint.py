r"""
Load a saved tiny character Transformer checkpoint and generate text.

Usage:

    D:\Anaconda3\python.exe generate_from_checkpoint.py
    D:\Anaconda3\python.exe generate_from_checkpoint.py "the model " 300
"""

import sys
from pathlib import Path

import torch

import day1_tiny_char_transformer as train_script


CHECKPOINT_PATH = Path(__file__).with_name("checkpoints") / "tiny_char_transformer.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run day1_tiny_char_transformer.py first to train and save it."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    config = checkpoint["config"]
    train_script.BLOCK_SIZE = config["block_size"]
    train_script.BATCH_SIZE = config["batch_size"]
    train_script.EMBED_DIM = config["embed_dim"]
    train_script.NUM_HEADS = config["num_heads"]
    train_script.NUM_LAYERS = config["num_layers"]
    train_script.DROPOUT = config["dropout"]
    train_script.CHARS = checkpoint["chars"]
    train_script.STOI = checkpoint["stoi"]
    train_script.ITOS = checkpoint["itos"]
    train_script.VOCAB_SIZE = config["vocab_size"]

    model = train_script.TinyCharTransformer().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def encode(prompt, stoi):
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(
            "Prompt contains characters outside the training vocabulary: "
            + repr("".join(unknown))
        )
    return torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=DEVICE)


def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "the model "
    max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 240

    model, checkpoint = load_model()
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    config = checkpoint["config"]
    final_losses = checkpoint.get("final_losses", {})

    print(f"device={DEVICE}")
    print(f"checkpoint={CHECKPOINT_PATH}")
    print(
        f"block_size={config['block_size']} "
        f"embed_dim={config['embed_dim']} "
        f"layers={config['num_layers']} "
        f"heads={config['num_heads']}"
    )
    if final_losses:
        print(
            f"saved_train_loss={final_losses['train']:.4f} "
            f"saved_val_loss={final_losses['val']:.4f}"
        )

    idx = encode(prompt, stoi)
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=max_new_tokens)[0].cpu()

    print()
    print("Generated text:")
    print(decode(generated, itos))


if __name__ == "__main__":
    main()
