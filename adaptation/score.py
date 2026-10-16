from __future__ import annotations

import torch

# every score is a per-sample estimate of log p_t(.) / p_s(.), positive = more target-like
# discrepancy for filtering is the negative, see filtering.py

@torch.no_grad()
def reverse_log_ratio(rev, state, action, cluster):
    return rev(state, action, cluster).squeeze(-1)

@torch.no_grad()
def dara_log_ratio(sas, sa, state, action, next_state):
    return (sas(state, action, next_state) - sa(state, action)).squeeze(-1)

@torch.no_grad()
def local_log_ratio(joint, state, action, cluster):
    return joint.dynamics_log_ratio(state, action, cluster).squeeze(-1)

@torch.no_grad()
def normalized_log_ratio(norm, state, action, cluster):
    return norm.dynamics_log_ratio(state, action, cluster).squeeze(-1)

@torch.no_grad()
def separate_log_ratio(zk, sa, state, action, cluster):
    return (zk(state, action, cluster) - sa(state, action)).squeeze(-1)
