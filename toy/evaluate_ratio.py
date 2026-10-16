from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

from toy.generate_dataset import make_case, HORIZON
from toy.linear_gaussian import true_dynamics_kl, true_next_state_log_ratio, true_cluster_log_ratio, true_reverse_log_ratio
from adaptation.clustering import quantile_edges, assign_bins
from adaptation.networks import train_domain_classifier
from adaptation.occupancy_classifier import OccupancyClassifier
from adaptation.sas_classifier import DynamicsClassifier
from adaptation.joint_cluster_classifier import JointClusterClassifier
from adaptation.reverse_classifier import ReverseClassifier, train_reverse_classifier
from adaptation.score import reverse_log_ratio, dara_log_ratio, local_log_ratio

# experiment 1: four-cell toy matrix
# does the reverse conditional (lodada) report dynamics shift when only occupancy shifted (case B)?
# does the occupancy-corrected local score track the true conditional ratio when both shift (case D)?

# helpers

def to_tensor(data, cluster):
    return tuple(torch.tensor(x, dtype = torch.float32) for x in data) + (torch.tensor(cluster),)

def corr(a, b):
    return np.nan if np.std(b) < 1e-8 else spearmanr(a, b).correlation

def fit_estimators(src, tar, *, num_clusters, steps, seed):
    torch.manual_seed(seed)
    ds, da = src[0].shape[-1], src[1].shape[-1]
    s, a, ns, c = src
    ts, ta, tns, tc = tar

    occ = train_domain_classifier(OccupancyClassifier(ds, da), (s, a), (ts, ta), steps = steps)
    sas = train_domain_classifier(DynamicsClassifier(ds, da), (s, a, ns), (ts, ta, tns), steps = steps)
    joint = train_domain_classifier(JointClusterClassifier(ds, da, num_clusters), (s, a, c), (ts, ta, tc), steps = steps)
    rev = train_reverse_classifier(ReverseClassifier(ds, da, num_clusters), (s, a, c), (ts, ta, tc), steps = steps)
    return occ, sas, joint, rev

def evaluate(name, src, tar, held, src_domain, tar_domain, *, num_clusters = 10, steps = 3000, seed = 0, horizon = HORIZON, s0_std = 1., s0_mean = (0., 0.)):
    edges = quantile_edges(tar.next_state, num_clusters)
    src_t = to_tensor(src, assign_bins(src.next_state, edges))
    tar_t = to_tensor(tar, assign_bins(tar.next_state, edges))
    held_c = assign_bins(held.next_state, edges)
    s, a, ns, c = to_tensor(held, held_c)

    occ, sas, joint, rev = fit_estimators(src_t, tar_t, num_clusters = num_clusters, steps = steps, seed = seed)

    l_rev = reverse_log_ratio(rev, s, a, c).numpy()
    l_dara = dara_log_ratio(sas, occ, s, a, ns).numpy()
    l_local = local_log_ratio(joint, s, a, c).numpy()

    kl = true_dynamics_kl(src_domain, tar_domain, held.state, held.action)
    true_next = true_next_state_log_ratio(src_domain, tar_domain, held.state, held.action, held.next_state)
    true_cluster = true_cluster_log_ratio(src_domain, tar_domain, held.state, held.action, held_c, edges)
    true_rev = true_reverse_log_ratio(src_domain, tar_domain, held.state, held.action, held.next_state, horizon = horizon, s0_std = s0_std, s0_mean = s0_mean)

    return dict(
        case = name,
        true_kl = kl.mean(),
        true_rev = np.abs(true_rev).mean(),
        abs_rev = np.abs(l_rev).mean(),
        abs_dara = np.abs(l_dara).mean(),
        abs_local = np.abs(l_local).mean(),
        rho_rev = corr(l_rev, true_cluster),
        rho_local = corr(l_local, true_cluster),
        rho_dara = corr(l_dara, true_next),
    )

def evaluate_case(name, *, num_clusters = 10, steps = 3000, seed = 0):
    src, tar, src_domain, tar_domain = make_case(name, seed = seed)
    held, *_ = make_case(name, seed = seed + 100)
    return evaluate(name, src, tar, held, src_domain, tar_domain, num_clusters = num_clusters, steps = steps, seed = seed)

def print_rows(rows):
    keys = [k for k in rows[0] if k != 'case']
    print('case  ' + ''.join(f'{k:>11}' for k in keys))
    for r in rows:
        print(f"{r['case']:<6}" + ''.join(f'{r[k]:>11.3f}' for k in keys))

def run(cases = 'ABCD', *, seeds = 1, **kwargs):
    rows = [evaluate_case(name, seed = seed, **kwargs) for seed in range(seeds) for name in cases]
    print_rows(rows)
    return rows

if __name__ == '__main__':
    run()
