from __future__ import annotations

import torch

# d_i = -log_ratio_i, w_i = exp(-alpha d_i)
# keep scores continuous for estimator validation, only threshold for downstream rl

def exp_weights(log_ratio, *, alpha = 1., max_weight = 1.):
    return (alpha * log_ratio).exp().clamp(max = max_weight)

def quantile_mask(log_ratio, *, keep = 0.5):
    return log_ratio >= torch.quantile(log_ratio, 1. - keep)
