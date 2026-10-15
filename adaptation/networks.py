from __future__ import annotations

import torch
from torch import nn, cat, Tensor
from torch.nn import Module
import torch.nn.functional as F

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def sample_batch(xs, batch_size):
    idx = torch.randint(0, xs[0].shape[0], (batch_size,))
    return tuple(x[idx] for x in xs)

def logits_of(out):
    return out if isinstance(out, tuple) else (out,)

# 2 x 256 mlp, matches the odrl dara classifier

class MLP(Module):
    def __init__(self, dim_in, dim_out = 1, *, dim_hidden = 256, depth = 2):
        super().__init__()
        dims = (dim_in, *((dim_hidden,) * depth))
        layers = []
        for i, o in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(i, o), nn.ReLU()])
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], dim_out))

    def forward(self, x):
        return self.net(x)

# balanced domain classifier training
# half source (label 0) half target (label 1) per minibatch, so logit ≈ log p_t(x) / p_s(x) without prior correction

def train_domain_classifier(clf, src, tar, *, steps = 3000, batch_size = 256, lr = 3e-4):
    device = next(clf.parameters()).device
    opt = torch.optim.Adam(clf.parameters(), lr = lr)
    half = batch_size // 2
    labels = cat((torch.zeros(half), torch.ones(half))).to(device)

    for _ in range(steps):
        inputs = tuple(cat((s, t)).to(device) for s, t in zip(sample_batch(src, half), sample_batch(tar, half)))
        loss = sum(F.binary_cross_entropy_with_logits(l.squeeze(-1), labels) for l in logits_of(clf(*inputs)))
        opt.zero_grad()
        loss.backward()
        opt.step()

    return clf
