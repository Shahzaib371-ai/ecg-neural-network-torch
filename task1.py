"""
Task 1 (1.1 - 1.7): Load and prepare MIT-BIH data
Run from VS Code:  python /home/shahzaibhasnain/Work/task1.py
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

DATA_DIR = "/home/shahzaibhasnain/Work/data"
CLASS_NAMES = ["N (Normal)", "S (Supraventricular)", "V (Ventricular)",
               "F (Fusion)", "Q (Unknown)"]

# 1.1 Load and print shapes
train_df = pd.read_csv(f"{DATA_DIR}/mitbih_train.csv", header=None)
test_df = pd.read_csv(f"{DATA_DIR}/mitbih_test.csv", header=None)
print("=== 1.1 Shapes ===")
print("Train:", train_df.shape, "| Test:", test_df.shape)

# 1.2 Separate signal and label, convert label to int
print("\n=== 1.2 Label type ===")
print("Label dtype before:", train_df[187].dtype)
X_all = train_df.iloc[:, :187].values
y_all = train_df[187].astype(int).values
X_test = test_df.iloc[:, :187].values
y_test = test_df[187].astype(int).values
print("Label dtype after :", y_all.dtype, "| unique:", np.unique(y_all))

# 1.3 Min / max signal value (train file)
print("\n=== 1.3 Signal range ===")
print("Min:", X_all.min(), "| Max:", X_all.max())

# 1.4 Rows per class
print("\n=== 1.4 Rows per class ===")
for k in range(5):
    print(f"{CLASS_NAMES[k]:22s} train: {(y_all == k).sum():6d}  "
          f"test: {(y_test == k).sum():6d}")

# 1.5 One beat per class
fig, axes = plt.subplots(1, 5, figsize=(16, 3), sharey=True)
t_ms = np.arange(187) * 1000 / 125          # 125 samples per second
for k, ax in enumerate(axes):
    ax.plot(t_ms, X_all[np.where(y_all == k)[0][0]])
    ax.set_title(CLASS_NAMES[k], fontsize=9)
    ax.set_xlabel("time (ms)")
axes[0].set_ylabel("amplitude")
fig.tight_layout()
fig.savefig("/home/shahzaibhasnain/Work/example_beats.png", dpi=150)
plt.close(fig)

# 1.6 Split (exact code from assignment)
rng = np.random.default_rng(42)
idx = rng.permutation(len(train_df))
n_val = int(0.1 * len(train_df))
val_idx, train_idx = idx[:n_val], idx[n_val:]

print("\n=== 1.6 Split sizes ===")
print("Train:", len(train_idx), "| Val:", len(val_idx), "| Test:", len(test_df))

# 1.7 Convert to tensors (float32 signals, long labels)
X_train = torch.tensor(X_all[train_idx], dtype=torch.float32)
y_train = torch.tensor(y_all[train_idx], dtype=torch.long)
X_val = torch.tensor(X_all[val_idx], dtype=torch.float32)
y_val = torch.tensor(y_all[val_idx], dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

print("\n=== 1.7 Tensors ===")
for name, t in [("X_train", X_train), ("y_train", y_train), ("X_val", X_val),
                ("y_val", y_val), ("X_test", X_test_t), ("y_test", y_test_t)]:
    print(f"{name:8s} shape={tuple(t.shape)}  dtype={t.dtype}")