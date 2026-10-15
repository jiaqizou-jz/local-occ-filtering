from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from adaptation.clustering import kmeans, assign
from adaptation.networks import train_domain_classifier
from adaptation.occupancy_classifier import OccupancyClassifier
from adaptation.sas_classifier import DynamicsClassifier
from adaptation.joint_cluster_classifier import JointClusterClassifier
from adaptation.reverse_classifier import ReverseClassifier, train_reverse_classifier
from adaptation.score import reverse_log_ratio, dara_log_ratio, local_log_ratio
from toy.evaluate_ratio import corr, false_negative_rate, false_positive_rate
from locomotion.data import load_transitions, split_episodes, subsample, Standardizer
from locomotion.truth import true_discrepancy

# experiments 2 and 3 on hopper
# 2: same physics, different behavior (source medium-replay, target expert): corrected score should sit closer to 0 than reverse
# 3: gravity shift + behavior shift: spearman of each score against the simulator ground truth D_true(s, a) on held-out source probes
# scores are log p_t / p_s estimates (positive = target-like), D_true is a discrepancy, so ranking metrics use -D_true as the truth

def f32(*xs):
    return tuple(torch.tensor(x, dtype = torch.float32) for x in xs)

def fit_estimators(src, tar, *, num_clusters, steps, batch_size, seed):
    torch.manual_seed(seed)
    s, a, ns, c = src
    ts, ta, tns, tc = tar
    ds, da = s.shape[-1], a.shape[-1]
    kw = dict(steps = steps, batch_size = batch_size)
    occ = train_domain_classifier(OccupancyClassifier(ds, da), (s, a), (ts, ta), **kw)
    sas = train_domain_classifier(DynamicsClassifier(ds, da), (s, a, ns), (ts, ta, tns), **kw)
    joint = train_domain_classifier(JointClusterClassifier(ds, da, num_clusters), (s, a, c), (ts, ta, tc), **kw)
    rev = train_reverse_classifier(ReverseClassifier(ds, da, num_clusters), (s, a, c), (ts, ta, tc), **kw)
    return occ, sas, joint, rev

@torch.no_grad()
def chunked(fn, *xs, size = 8192):
    return torch.cat([fn(*(x[i:i + size] for x in xs)) for i in range(0, len(xs[0]), size)]).numpy()

def evaluate(source_path, target_path, *, gravity_target, num_clusters, steps, batch_size, seed, target_budget, held_fraction):
    src_all, tar = load_transitions(source_path), load_transitions(target_path)
    tar = subsample(tar, target_budget, seed = seed)
    src, held = split_episodes(src_all, held_fraction = held_fraction, seed = seed)

    std = Standardizer(src)
    s, a, ns = std(src)
    ts, ta, tns = std(tar)
    hs, ha, hns = std(held)

    centroids = kmeans(tns, num_clusters, seed = seed)
    c, tc, hc = (torch.tensor(assign(x, centroids)) for x in (ns, tns, hns))

    occ, sas, joint, rev = fit_estimators(f32(s, a, ns) + (c,), f32(ts, ta, tns) + (tc,), num_clusters = num_clusters, steps = steps, batch_size = batch_size, seed = seed)

    hs_t, ha_t, hns_t = f32(hs, ha, hns)
    l_rev = chunked(lambda *x: reverse_log_ratio(rev, *x), hs_t, ha_t, hc)
    l_dara = chunked(lambda *x: dara_log_ratio(sas, occ, *x), hs_t, ha_t, hns_t)
    l_local = chunked(lambda *x: local_log_ratio(joint, *x), hs_t, ha_t, hc)
    l_occ = chunked(lambda *x: occ(*x).squeeze(-1), hs_t, ha_t)

    row = dict(
        seed = seed, n_src = len(s), n_tar = len(ts), n_held = len(hs), cluster_sizes_min = int(np.bincount(tc.numpy(), minlength = num_clusters).min()),
        mean_rev = l_rev.mean(), mean_dara = l_dara.mean(), mean_local = l_local.mean(), mean_occ = l_occ.mean(),
        abs_rev = np.abs(l_rev).mean(), abs_dara = np.abs(l_dara).mean(), abs_local = np.abs(l_local).mean(), abs_occ = np.abs(l_occ).mean(),
        p90_rev = np.quantile(np.abs(l_rev), .9), p90_dara = np.quantile(np.abs(l_dara), .9), p90_local = np.quantile(np.abs(l_local), .9),
    )

    if gravity_target != 1.:
        d_true, ns_src, ns_tar = true_discrepancy(held.state, held.action, gravity_target = gravity_target, next_state_std = std.std[2])
        truth = -d_true
        c_src = assign((ns_src - std.mean[2]) / std.std[2], centroids)
        c_tar = assign((ns_tar - std.mean[2]) / std.std[2], centroids)
        row.update(
            d_true_mean = d_true.mean(), cluster_flip = (c_src != c_tar).mean(),
            rho_rev = corr(l_rev, truth), rho_local = corr(l_local, truth), rho_dara = corr(l_dara, truth), rho_occ = corr(l_occ, truth),
            fn_rev = false_negative_rate(l_rev, truth), fn_local = false_negative_rate(l_local, truth), fn_dara = false_negative_rate(l_dara, truth),
            fp_rev = false_positive_rate(l_rev, truth), fp_local = false_positive_rate(l_local, truth), fp_dara = false_positive_rate(l_dara, truth),
        )
    return row, dict(rev = l_rev, dara = l_dara, local = l_local, occ = l_occ, d_true = d_true if gravity_target != 1. else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required = True)
    ap.add_argument('--target', required = True)
    ap.add_argument('--gravity_target', type = float, default = 1., help = '1.0 = same physics (experiment 2), else the target gravity scale (experiment 3)')
    ap.add_argument('--num_clusters', type = int, default = 10)
    ap.add_argument('--steps', type = int, default = 10000)
    ap.add_argument('--batch_size', type = int, default = 256)
    ap.add_argument('--seeds', type = int, default = 1)
    ap.add_argument('--first_seed', type = int, default = 0)
    ap.add_argument('--target_budget', type = int, default = None)
    ap.add_argument('--held_fraction', type = float, default = 0.2)
    ap.add_argument('--out', default = None, help = 'csv path for per-seed rows')
    ap.add_argument('--save_scores', default = None, help = 'npz path for the held-out scores of the last seed')
    args = ap.parse_args()

    rows = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        row, scores = evaluate(args.source, args.target, gravity_target = args.gravity_target, num_clusters = args.num_clusters, steps = args.steps,
                               batch_size = args.batch_size, seed = seed, target_budget = args.target_budget, held_fraction = args.held_fraction)
        rows.append(row)
        print(' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()), flush = True)
    if args.save_scores:
        np.savez(args.save_scores, **{k: v for k, v in scores.items() if v is not None})
    if args.out:
        with open(args.out, 'w', newline = '') as f:
            w = csv.DictWriter(f, fieldnames = list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__ == '__main__':
    main()
