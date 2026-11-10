"""
Task 3.2 - Gradient check

Compares the output-layer gradients computed by PyTorch autograd with the
ones computed by hand:
    dJ/dW[3] = (1/B) (Yhat - Y)^T A[2]
    dJ/db[3] = (1/B) sum_i (yhat_i - y_i)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from task2 import ECGNet, cross_entropy


def load_train(data_dir="data"):
    """Read mitbih_train.csv exactly as stored (no header row)."""
    path = Path(data_dir) / "mitbih_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Download the dataset first.")
    return pd.read_csv(path, header=None)


def main():
    # ---- Load data and make the train split exactly as in Task 1.6 ----
    train_df = load_train()

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_df))
    n_val = int(0.1 * len(train_df))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_part = train_df.iloc[train_idx]
    X = torch.tensor(train_part.iloc[:, :187].values, dtype=torch.float32)
    y = torch.tensor(train_part.iloc[:, 187].values.astype(int), dtype=torch.long)

    # ---- First 64 training rows ----
    xb, yb = X[:64], y[:64]
    B = xb.shape[0]

    # ---- Fresh network (seed 0) ----
    torch.manual_seed(0)
    model = ECGNet()

    # ---- Forward pass, loss, and autograd gradients ----
    logits = model(xb)
    loss = cross_entropy(logits, yb)
    print(f"Initial loss: {loss.item():.4f}   (log 5 = {np.log(5):.4f})")

    model.zero_grad()
    loss.backward()
    grad_W3_torch = model.W[2].grad.clone()
    grad_b3_torch = model.b[2].grad.clone()

    # ---- Hand-computed gradients ----
    with torch.no_grad():
        # A[2]: activations of the second hidden layer, shape (B, 64)
        a1 = model.relu(xb @ model.W[0].T + model.b[0])
        a2 = model.relu(a1 @ model.W[1].T + model.b[1])

        # Yhat: softmax probabilities, shape (B, 5)
        y_hat = torch.softmax(logits, dim=1)

        # Y: one-hot labels, shape (B, 5)
        Y = torch.zeros_like(y_hat)
        Y.scatter_(1, yb.unsqueeze(1), 1.0)

        grad_W3_hand = (y_hat - Y).T @ a2 / B       # shape (5, 64)
        grad_b3_hand = (y_hat - Y).sum(dim=0) / B   # shape (5,)

    # ---- Largest absolute differences ----
    diff_W3 = (grad_W3_torch - grad_W3_hand).abs().max().item()
    diff_b3 = (grad_b3_torch - grad_b3_hand).abs().max().item()

    print(f"Max |difference| for W[3]: {diff_W3:.3e}")
    print(f"Max |difference| for b[3]: {diff_b3:.3e}")
    print("Check passed (< 1e-5):", diff_W3 < 1e-5 and diff_b3 < 1e-5)


if __name__ == "__main__":
    main()