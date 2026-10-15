from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

from toy.generate_dataset import make_case
from toy.linear_gaussian import true_dynamics_kl, true_next_state_log_ratio, true_cluster_log_ratio, true_reverse_log_ratio
from adaptation.clustering import quantile_edges, assign_bins
from adaptation.networks import train_domain_classifier
from adaptation.occupancy_classifier import OccupancyClassifier
from adaptation.sas_classifier import DynamicsClassifier
from adaptation.joint_cluster_classifier import JointClusterClassifier
from adaptation.reverse_classifier import ReverseClassifier, train_reverse_classifier
from adaptation.score import reverse_log_ratio, dara_log_ratio, local_log_ratio

# experiment 1: four-cell toy matrix, plus the adversarial case E of experiment 1b
# does the reverse conditional (lodada) report dynamics shift when only occupancy shifted (case B)?
# does the occupancy-corrected local score track the true conditional ratio when both shift (case D)?

# helpers

def to_tensor(data, cluster):
    return tuple(torch.tensor(x, dtype = torch.float32) for x in data) + (torch.tensor(cluster),)

def corr(a, b):
    return np.nan if np.std(b) < 1e-8 else spearmanr(a, b).correlation

# per-sample filtering errors, discrepancy d = -log ratio
# false negative: a transition in the top `top` of true discrepancy that the score ranks in its bottom `bottom`
# false positive: a transition in the bottom `bottom` of true discrepancy that the score ranks in its top `top`

def false_negative_rate(score, truth, *, top = 0.25, bottom = 0.5):
    if np.std(truth) < 1e-8:
        return np.nan
    d_true, d_score = -truth, -score
    shifted = d_true >= np.quantile(d_true, 1 - top)
    missed = d_score <= np.quantile(d_score, bottom)
    return (shifted & missed).sum() / shifted.sum()

def false_positive_rate(score, truth, *, top = 0.25, bottom = 0.5):
    if np.std(truth) < 1e-8:
        return np.nan
    d_true, d_score = -truth, -score
    fine = d_true <= np.quantile(d_true, bottom)
    flagged = d_score >= np.quantile(d_score, 1 - top)
    return (fine & flagged).sum() / fine.sum()

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

def evaluate_case(name, *, num_clusters = 10, steps = 3000, seed = 0, horizon = 50):
    src, tar, src_domain, tar_domain = make_case(name, seed = seed, horizon = horizon)
    held, *_ = make_case(name, seed = seed + 100, horizon = horizon)

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
    # what a perfect reverse classifier would output: dyn + occ - marginal, all analytic
    true_rev = true_reverse_log_ratio(src_domain, tar_domain, held.state, held.action, held_c, edges, horizon)

    return dict(
        case = name,
        true_kl = kl.mean(),
        abs_rev = np.abs(l_rev).mean(),
        abs_dara = np.abs(l_dara).mean(),
        abs_local = np.abs(l_local).mean(),
        abs_rev_true = np.abs(true_rev).mean(),
        rho_rev = corr(l_rev, true_cluster),
        rho_local = corr(l_local, true_cluster),
        rho_dara = corr(l_dara, true_next),
        rho_rev_true = corr(true_rev, true_cluster),
        fn_rev = false_negative_rate(l_rev, true_cluster),
        fn_local = false_negative_rate(l_local, true_cluster),
        fn_dara = false_negative_rate(l_dara, true_next),
        fn_rev_true = false_negative_rate(true_rev, true_cluster),
        fp_rev = false_positive_rate(l_rev, true_cluster),
        fp_local = false_positive_rate(l_local, true_cluster),
        fp_dara = false_positive_rate(l_dara, true_next),
        fp_rev_true = false_positive_rate(true_rev, true_cluster),
    )

def run(cases = 'ABCDE', *, seeds = 1, first_seed = 0, out = None, **kwargs):
    rows = [dict(seed = seed, **evaluate_case(name, seed = seed, **kwargs)) for seed in range(first_seed, first_seed + seeds) for name in cases]
    keys = [k for k in rows[0] if k not in ('case', 'seed')]
    print('seed  case  ' + ''.join(f'{k:>13}' for k in keys))
    for r in rows:
        print(f"{r['seed']:<6}{r['case']:<6}" + ''.join(f'{r[k]:>13.3f}' for k in keys))
    if seeds > 1:
        print('\nmean +- std over seeds')
        print('case  ' + ''.join(f'{k:>19}' for k in keys))
        for name in cases:
            sub = [r for r in rows if r['case'] == name]
            print(f'{name:<6}' + ''.join(f"{np.mean([r[k] for r in sub]):>9.3f} +- {np.std([r[k] for r in sub]):<5.3f}" for k in keys))
    if out is not None:
        import csv
        with open(out, 'w', newline = '') as f:
            w = csv.DictWriter(f, fieldnames = ['seed', 'case'] + keys)
            w.writeheader()
            w.writerows(rows)
    return rows

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default = 'ABCDE')
    ap.add_argument('--seeds', type = int, default = 1)
    ap.add_argument('--first_seed', type = int, default = 0)
    ap.add_argument('--num_clusters', type = int, default = 10)
    ap.add_argument('--steps', type = int, default = 3000)
    ap.add_argument('--out', default = None, help = 'csv path for per-seed rows')
    args = ap.parse_args()
    run(args.cases, seeds = args.seeds, first_seed = args.first_seed, num_clusters = args.num_clusters, steps = args.steps, out = args.out)
