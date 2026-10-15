from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import spearmanr

from toy.generate_dataset import make_case, SOURCE, CASES
from toy.linear_gaussian import rollout, true_cluster_log_ratio, true_occupancy_log_ratio, true_marginal_log_ratio
from toy.evaluate_ratio import false_negative_rate, false_positive_rate
from adaptation.clustering import quantile_edges, assign_bins

# experiment 1b design: search target behaviors for a false negative of the reverse score, on analytic ratios only
# l_rev = l_dyn + l_occ - l_marg, every term exact for the 1d linear gaussian rollout (toy/linear_gaussian.py)
# evaluated on held-out source transitions, clusters are target-fit quantile bins as in evaluate_ratio
# run with: python -m toy.design_case_e

HORIZON = 50

def analytic_reverse(tar_domain, held, *, num_clusters = 10):
    tar = rollout(tar_domain, num_traj = 100, horizon = HORIZON, seed = 1)
    edges = quantile_edges(tar.next_state, num_clusters)
    c = assign_bins(held.next_state, edges)
    l_dyn = true_cluster_log_ratio(SOURCE, tar_domain, held.state, held.action, c, edges)
    l_rev = (l_dyn + true_occupancy_log_ratio(SOURCE, tar_domain, held.state, held.action, HORIZON)
             - true_marginal_log_ratio(SOURCE, tar_domain, c, edges, HORIZON))
    return dict(
        abs_dyn = np.abs(l_dyn).mean(),
        abs_rev = np.abs(l_rev).mean(),
        rho = spearmanr(l_rev, l_dyn).correlation,
        fn = false_negative_rate(l_rev, l_dyn),
        fp = false_positive_rate(l_rev, l_dyn),
    )

def show(label, r):
    print(f'{label:<40}' + ''.join(f'{k} {v:7.3f}  ' for k, v in r.items()))

if __name__ == '__main__':
    held, *_ = make_case('A', seed = 100)
    dyn = CASES['C']

    print('# magnitude: can any target behavior drive E|l_rev| below E|l_dyn|?  (C dynamics, grid over K and sigma_a)')
    rows = []
    for K, sa in itertools.product(np.round(np.arange(-1.6, 0.45, 0.2), 2), [0.1, 0.15, 0.2, 0.3, 0.45, 0.6]):
        r = analytic_reverse(dyn._replace(K = K, sigma_a = sa), held)
        rows.append((r['abs_rev'] / r['abs_dyn'], K, sa, r))
    for ratio, K, sa, r in sorted(rows, key = lambda x: x[0])[:5]:
        show(f'K {K:5.2f} sigma_a {sa:4.2f} ratio {ratio:5.2f}', r)

    print('\n# ranking: target behavior noise sweep (C dynamics, K = source)')
    for sa in [0.3, 0.2, 0.15, 0.1, 0.05]:
        show(f'sigma_a {sa:4.2f}', analytic_reverse(dyn._replace(sigma_a = sa), held))

    print('\n# chosen case E')
    show('E', analytic_reverse(CASES['E'], held))
