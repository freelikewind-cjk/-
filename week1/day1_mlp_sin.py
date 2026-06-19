"""
Day 1: train a tiny neural network to learn y = sin(x).

You do not need to understand every line today.
Your first goal is only this:

1. Run the file.
2. See loss go down.
3. Change a few numbers and observe what happens.
"""

import math

import torch
from torch import nn


# Try changing these three numbers first.
LEARNING_RATE = 0.005
HIDDEN_SIZE = 18
STEPS = 2000


def make_data():
    """Create toy training data: x -> sin(x)."""
    x = torch.linspace(-2 * math.pi, 2 * math.pi, 256).unsqueeze(1)
    y = torch.sin(x)
    return x, y


class TinyMLP(nn.Module):
    """A small 2-layer neural network."""

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


def train():
    x, y = make_data()

    model = TinyMLP()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(STEPS + 1):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step={step:4d} loss={loss.item():.6f} lr={LEARNING_RATE}")

    print()
    print("Done. Now try changing LEARNING_RATE, HIDDEN_SIZE, or STEPS.")


if __name__ == "__main__":
    train()

