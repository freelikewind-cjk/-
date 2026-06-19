"""
Day 3: overfitting and weight decay.

This file still learns y = sin(x), but the training data is intentionally
small and noisy. That makes it easier to see overfitting:

1. train_loss can become low because the model memorizes noisy training data.
2. val_loss tells us whether the model works on clean held-out data.
3. WEIGHT_DECAY is one simple regularization knob.

Your goal today:

1. Run the file.
2. Change HIDDEN_SIZE, TRAIN_SIZE, NOISE_STD, WEIGHT_DECAY.
3. Compare train_loss and val_loss.
"""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LEARNING_RATE = 0.01
BATCH_SIZE = 16
HIDDEN_SIZE = 128
STEPS = 3000
EVAL_EVERY = 200
TRAIN_SIZE = 48
NOISE_STD = 0.20
WEIGHT_DECAY = 0.0


def make_data():
    generator = torch.Generator().manual_seed(0)

    train_x = torch.linspace(-2 * math.pi, 2 * math.pi, TRAIN_SIZE).unsqueeze(1)
    clean_train_y = torch.sin(train_x)
    noise = torch.randn(clean_train_y.shape, generator=generator) * NOISE_STD
    train_y = clean_train_y + noise

    val_x = torch.linspace(-2 * math.pi, 2 * math.pi, 256).unsqueeze(1)
    val_y = torch.sin(val_x)

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
    torch.manual_seed(0)

    train_data, val_data = make_data()
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = TinyMLP()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

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
            gap = val_loss - train_loss.item()
            print(
                f"step={step:4d} "
                f"train_loss={train_loss.item():.6f} "
                f"val_loss={val_loss:.6f} "
                f"gap={gap:.6f} "
                f"hidden={HIDDEN_SIZE} "
                f"train_size={TRAIN_SIZE} "
                f"noise={NOISE_STD} "
                f"weight_decay={WEIGHT_DECAY}"
            )

    print()
    print("Done. Try WEIGHT_DECAY=0.0, 0.001, 0.01 and compare val_loss.")


if __name__ == "__main__":
    train()
