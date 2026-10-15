from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from toy.generate_dataset import make_case
from toy.linear_gaussian import true_next_state_log_ratio, true_cluster_log_ratio
from adaptation.clustering import quantile_edges, assign_bins
from toy.evaluate_ratio import to_tensor, fit_estimators
from adaptation.score import dara_log_ratio, local_log_ratio

# experiment 0: joint ratio - occupancy ratio = conditional ratio, checked on the estimators themselves
# spearman, not pearson: the true cluster ratio has a clipped tail near -27 where the classifier logit cannot follow
# run with: python -m pytest toy/test_ratio_identity.py

def test_joint_minus_occupancy_is_conditional():
    src, tar, src_domain, tar_domain = make_case('D', seed = 0)
    held, *_ = make_case('D', seed = 100)
    num_clusters = 10

    edges = quantile_edges(tar.next_state, num_clusters)
    src_t, tar_t = to_tensor(src, assign_bins(src.next_state, edges)), to_tensor(tar, assign_bins(tar.next_state, edges))
    held_c = assign_bins(held.next_state, edges)
    s, a, ns, c = to_tensor(held, held_c)

    occ, sas, joint, _ = fit_estimators(src_t, tar_t, num_clusters = num_clusters, steps = 3000, seed = 0)

    true_next = true_next_state_log_ratio(src_domain, tar_domain, held.state, held.action, held.next_state)
    true_cluster = true_cluster_log_ratio(src_domain, tar_domain, held.state, held.action, held_c, edges)

    assert spearmanr(dara_log_ratio(sas, occ, s, a, ns).numpy(), true_next).correlation > 0.9
    assert spearmanr(local_log_ratio(joint, s, a, c).numpy(), true_cluster).correlation > 0.9
