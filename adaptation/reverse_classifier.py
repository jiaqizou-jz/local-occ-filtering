from __future__ import annotations

import torch
from torch import cat, stack
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from adaptation.networks import MLP

# lodada-style reverse conditional baseline
# one classifier per cluster, D_k(s, a) = P(target | s, a, C = k)
# trained with balanced source / target batches inside each cluster, so logit ≈ log p_t(s, a | C) / p_s(s, a | C)

class ReverseClassifier(Module):
    def __init__(self, dim_state, dim_action, num_clusters, *, dim_hidden = 256):
        super().__init__()
        self.heads = ModuleList([MLP(dim_state + dim_action, dim_hidden = dim_hidden) for _ in range(num_clusters)])

    def forward(self, state, action, cluster):
        sa = cat((state, action), dim = -1)
        logits = stack([head(sa) for head in self.heads], dim = 1)
        return logits.gather(1, cluster[:, None, None]).squeeze(1)

def train_reverse_classifier(clf, src, tar, *, steps = 3000, batch_size = 256, lr = 3e-4):
    device = next(clf.parameters()).device
    num_clusters = len(clf.heads)
    src_idx = [torch.where(src[2] == k)[0] for k in range(num_clusters)]
    tar_idx = [torch.where(tar[2] == k)[0] for k in range(num_clusters)]
    opt = torch.optim.Adam(clf.parameters(), lr = lr)
    half = batch_size // 2
    labels = cat((torch.zeros(half), torch.ones(half))).to(device)

    for _ in range(steps):
        loss = 0.
        for si, ti in zip(src_idx, tar_idx):
            if len(si) == 0 or len(ti) == 0:
                continue
            si = si[torch.randint(0, len(si), (half,))]
            ti = ti[torch.randint(0, len(ti), (half,))]
            inputs = tuple(cat((s[si], t[ti])).to(device) for s, t in zip(src, tar))
            loss = loss + F.binary_cross_entropy_with_logits(clf(*inputs).squeeze(-1), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return clf
