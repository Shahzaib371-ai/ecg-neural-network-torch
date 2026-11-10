"""
Task 2 - Build the network
Task 3.1 - Softmax cross-entropy loss (at the bottom of this file)

Architecture (as required by the assignment):
    Layer 1 (hidden) : 187 -> 128, ReLU
    Layer 2 (hidden) : 128 -> 64,  ReLU
    Layer 3 (output) : 64  -> 5,   no activation (logits)

Notation: layer l has weight matrix W[l] of shape (n[l], n[l-1]) and bias b[l].
    z[l] = W[l] a[l-1] + b[l],   a[l] = g(z[l]),   a[0] = x

Only allowed tools are used here: nn.Module, nn.Parameter, nn.ParameterList
and basic torch tensor operations. No nn.Linear, nn.Sequential, nn.ReLU or
nn.init.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 2.1 The ECGNet class: every W[l] and b[l] is created as an nn.Parameter
# ---------------------------------------------------------------------------
class ECGNet(nn.Module):
    def __init__(self, layer_sizes=(187, 128, 64, 5)):
        super().__init__()
        self.layer_sizes = layer_sizes

        # ParameterLists make sure PyTorch registers every tensor as a
        # trainable parameter of the model.
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()

        # Create one (W, b) pair for each pair of consecutive layer sizes.
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):

            # -----------------------------------------------------------
            # 2.2 He initialisation
            #     W[l]_ij ~ N(0, sqrt(2 / n[l-1]))
            #     (the second argument is the standard deviation)
            #     b[l] = 0
            # -----------------------------------------------------------
            std = math.sqrt(2.0 / n_in)
            W = torch.randn(n_out, n_in) * std    # shape (n[l], n[l-1])
            b = torch.zeros(n_out)                # shape (n[l],)

            self.W.append(nn.Parameter(W))
            self.b.append(nn.Parameter(b))

    # -----------------------------------------------------------------------
    # ReLU written by hand: g(z) = max(z, 0)
    # -----------------------------------------------------------------------
    @staticmethod
    def relu(z):
        return torch.clamp(z, min=0)

    # -----------------------------------------------------------------------
    # 2.3 Forward pass for a mini-batch x of shape (B, 187)
    # -----------------------------------------------------------------------
    def forward(self, x):
        a = x                          # a[0] = x
        last = len(self.W) - 1         # index of the output layer

        for l in range(len(self.W)):
            # With one example per row the layer becomes
            #     Z[l] = A[l-1] W[l]^T + b[l]
            z = a @ self.W[l].T + self.b[l]

            if l < last:
                a = self.relu(z)       # hidden layers use ReLU
            else:
                a = z                  # output layer returns raw logits

        return a                       # logits z[3], shape (B, 5)

    # -----------------------------------------------------------------------
    # 2.4 Count the trainable parameters
    # -----------------------------------------------------------------------
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 3.1 Softmax cross-entropy
#     L = -z_y + log( sum_j exp(z_j) )
#     The logsumexp form is used because computing exp(z) directly can overflow.
# ---------------------------------------------------------------------------
def cross_entropy(logits, y):
    """logits: (B, 5) float tensor, y: (B,) long tensor with values 0..4.
    Returns the mean loss over the batch (a scalar tensor)."""
    lse = torch.logsumexp(logits, dim=1)                   # log sum_j exp(z_j), shape (B,)
    true_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)   # z_y for each row, shape (B,)
    return (lse - true_logit).mean()


# ---------------------------------------------------------------------------
# Quick self-test: run with   python src/model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 2.2 Seed must be set before the network is created
    torch.manual_seed(0)
    model = ECGNet()

    # 2.4 Number of trainable parameters
    print("Trainable parameters:", model.count_parameters())

    # Hand calculation (to be shown in the report):
    #   Layer 1: 187 * 128 + 128 = 24,064
    #   Layer 2: 128 * 64  + 64  =  8,256
    #   Layer 3: 64  * 5   + 5   =    325
    #   Total                    = 32,645

    # 2.3 Check the output shape of the forward pass
    x = torch.randn(4, 187)
    print("Output shape:", model(x).shape)    # expected: torch.Size([4, 5])