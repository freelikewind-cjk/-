"""
Day 2: train/val split + mini-batch training.

This file is still learning y = sin(x), but it is closer to real training code:

1. Training data and validation data are separated.
2. DataLoader gives the model one mini-batch at a time.
3. We check validation loss regularly.

Your goal today:

1. Run the file.
2. Change LEARNING_RATE, BATCH_SIZE, HIDDEN_SIZE.
3. Watch both train_loss and val_loss.
"""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LEARNING_RATE = 0.01
BATCH_SIZE = 32
HIDDEN_SIZE = 64
STEPS = 2000
EVAL_EVERY = 100


def make_data():
    x = torch.linspace(-2 * math.pi, 2 * math.pi, 512).unsqueeze(1)
    y = torch.sin(x)

    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(x.shape[0], generator=generator)
    train_indices = indices[:400]
    val_indices = indices[400:]

    train_x = x[train_indices]
    train_y = y[train_indices]
    val_x = x[val_indices]
    val_y = y[val_indices]

    train_data = TensorDataset(train_x, train_y)
    val_data = TensorDataset(val_x, val_y)
    return train_data, val_data


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, val_loader, loss_fn):
    model.eval()

    total_loss = 0.0
    total_items = 0

    for x, y in val_loader:
        pred = model(x)
        loss = loss_fn(pred, y)

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size

    model.train()
    return total_loss / total_items


def train():
    train_data, val_data = make_data()

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = TinyMLP()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_iter = iter(train_loader)

    for step in range(STEPS + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        pred = model(x)
        train_loss = loss_fn(pred, y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if step % EVAL_EVERY == 0:
            val_loss = evaluate(model, val_loader, loss_fn)
            print(
                f"step={step:4d} "
                f"train_loss={train_loss.item():.6f} "
                f"val_loss={val_loss:.6f} "
                f"lr={LEARNING_RATE} "
                f"batch_size={BATCH_SIZE}"
            )

    print()
    print("Done. Try BATCH_SIZE=8, 32, 128 and compare train/val loss.")


if __name__ == "__main__":
    train()
