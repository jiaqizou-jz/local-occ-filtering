from __future__ import annotations

import time

import numpy as np
import torch
from scipy.stats import spearmanr

from adaptation.clustering import kmeans, assign
from adaptation.networks import train_domain_classifier
from adaptation.occupancy_classifier import OccupancyClassifier
from adaptation.sas_classifier import DynamicsClassifier
from adaptation.joint_cluster_classifier import JointClusterClassifier
from adaptation.reverse_classifier import ReverseClassifier, train_reverse_classifier
from adaptation.normalized_cluster_classifier import ClusterPredictor, train_cluster_predictor, NormalizedClusterClassifier, ClusterConditionedClassifier
from adaptation.score import reverse_log_ratio, dara_log_ratio, local_log_ratio, normalized_log_ratio, separate_log_ratio
from odrl.datasets import Transitions, load_d4rl, load_odrl, split, subsample
from odrl.ground_truth import make_source_env, make_target_env, dynamics_gap, predict

# experiments 2 and 3 on hopper
# source = one d4rl hopper set (standard physics), target = 5000 transitions
#   null      another 5000 transitions of the source set       -> no shift at all, floor for abs_*
#   behavior  d4rl hopper-expert-v2                            -> same physics, behavior shift only (experiment 2)
#   g<x>_<q>  odrl hopper-gravity x, quality q                  -> dynamics + behavior shift (experiment 3)
# scores are evaluated on held-out source transitions; truth for ranking = simulator next-state gap (0 for null / behavior)
# occ = logit D_SA alone, the occupancy-only control: an estimator that only matches rho_occ is ranking by occupancy, not dynamics
# experiment 2 passes if abs_local(behavior) is closer to abs_local(null) than abs_rev(behavior) is to abs_rev(null)
# experiment 3 passes if rho_local > rho_rev on every gravity row

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# prefix = hopper / halfcheetah / walker2d, axis = gravity / friction: source {prefix}-medium-replay-v2, env {prefix}-{axis}
# behavior_m = d4rl {prefix}-medium-v2, a second same-physics control with more overlap than expert
# paired_<g> = the null chunk's own (s, a) stepped in the target physics: exact source occupancy, pure dynamics shift
#   (the fixed probe distribution of the plan: support is perfect by construction, so only resolution and power remain)

def cases_for(prefix):
    return dict(
        null = None,
        behavior = ('d4rl', f'{prefix}-expert-v2'),
        behavior_m = ('d4rl', f'{prefix}-medium-v2'),
        **{f'g{g}_{q}': ('odrl', g, q) for g in (0.5, 2.0) for q in ('expert', 'medium', 'random')},
        **{f'paired_{g}': ('paired', g) for g in (0.5, 2.0)}
    )

CASES = cases_for('hopper')
SOURCE = 'hopper-medium-replay-v2'
ENV = 'hopper-gravity'

# helpers

def to_tensor(data, cluster):
    return tuple(torch.tensor(x) for x in data) + (torch.tensor(cluster),)

def corr(a, b):
    return np.nan if np.std(b) < 1e-8 else spearmanr(a, b).correlation

def batched(fn, *xs, device, chunk = 8192):
    with torch.no_grad():
        return torch.cat([fn(*(x[i:i + chunk].to(device) for x in xs)).cpu() for i in range(0, len(xs[0]), chunk)]).numpy()

def fit_estimators(src, tar, *, num_clusters, steps, seed, device):
    torch.manual_seed(seed)
    ds, da = src[0].shape[-1], src[1].shape[-1]
    s, a, ns, c = src
    ts, ta, tns, tc = tar

    occ = train_domain_classifier(OccupancyClassifier(ds, da).to(device), (s, a), (ts, ta), steps = steps)
    sas = train_domain_classifier(DynamicsClassifier(ds, da).to(device), (s, a, ns), (ts, ta, tns), steps = steps)
    joint = train_domain_classifier(JointClusterClassifier(ds, da, num_clusters).to(device), (s, a, c), (ts, ta, tc), steps = steps)
    rev = train_reverse_classifier(ReverseClassifier(ds, da, num_clusters).to(device), (s, a, c), (ts, ta, tc), steps = steps)
    pred = train_cluster_predictor(ClusterPredictor(ds, da, num_clusters).to(device), (s, a, c), steps = steps)
    norm = train_domain_classifier(NormalizedClusterClassifier(ds, da, num_clusters, pred).to(device), (s, a, c), (ts, ta, tc), steps = steps)
    zk = train_domain_classifier(ClusterConditionedClassifier(ds, da, num_clusters).to(device), (s, a, c), (ts, ta, tc), steps = steps)
    return occ, sas, joint, rev, norm, zk

def load_target(name, null, *, num_target, seed, prefix = 'hopper', axis = 'gravity'):
    spec = cases_for(prefix)[name]
    if spec is None:
        return null, None
    if spec[0] == 'd4rl':
        return subsample(load_d4rl(spec[1]), num_target, seed = seed), None
    if spec[0] == 'paired':
        env = make_target_env(f'{prefix}-{axis}', spec[1])
        return Transitions(null.state, null.action, predict(env, null.state, null.action).astype(np.float32)), env
    _, shift_level, quality = spec
    return load_odrl(f'{prefix}-{axis}', shift_level, quality), make_target_env(f'{prefix}-{axis}', shift_level)

def evaluate_case(name, *, prefix = 'hopper', axis = 'gravity', num_clusters = 10, steps = 3000, num_target = 5000, num_held = 50_000, seed = 0, device = DEVICE):
    train, held, null = split(load_d4rl(f'{prefix}-medium-replay-v2'), num_held = num_held, num_null = num_target, seed = seed)
    tar, tar_env = load_target(name, null, num_target = num_target, seed = seed, prefix = prefix, axis = axis)
    src_env = make_source_env(f'{prefix}-{axis}')

    # clusters on standardized target next states, shared by both domains
    scale = tar.next_state.std(0) + 1e-6
    centroids = kmeans(tar.next_state / scale, num_clusters, seed = seed)
    cluster_of = lambda x: assign(x / scale, centroids)
    src_t = to_tensor(train, cluster_of(train.next_state))
    tar_t = to_tensor(tar, cluster_of(tar.next_state))
    s, a, ns, c = to_tensor(held, cluster_of(held.next_state))

    occ, sas, joint, rev, norm, zk = fit_estimators(src_t, tar_t, num_clusters = num_clusters, steps = steps, seed = seed, device = device)

    l_rev = batched(lambda s, a, c: reverse_log_ratio(rev, s, a, c), s, a, c, device = device)
    l_dara = batched(lambda s, a, ns: dara_log_ratio(sas, occ, s, a, ns), s, a, ns, device = device)
    l_local = batched(lambda s, a, c: local_log_ratio(joint, s, a, c), s, a, c, device = device)
    l_norm = batched(lambda s, a, c: normalized_log_ratio(norm, s, a, c), s, a, c, device = device)
    l_sep = batched(lambda s, a, c: separate_log_ratio(zk, occ, s, a, c), s, a, c, device = device)
    l_occ = batched(lambda s, a: occ(s, a).squeeze(-1), s, a, device = device)

    gap, f_src = dynamics_gap(src_env, tar_env if tar_env is not None else src_env, held.state, held.action, scale = scale)
    recon = np.linalg.norm((f_src - held.next_state) / scale, axis = -1)   # source simulator vs the data's own s'

    return dict(
        case = name,
        gap = gap.mean(),
        recon = np.median(recon),
        abs_rev = np.abs(l_rev).mean(),
        abs_dara = np.abs(l_dara).mean(),
        abs_local = np.abs(l_local).mean(),
        abs_norm = np.abs(l_norm).mean(),
        abs_sep = np.abs(l_sep).mean(),
        abs_occ = np.abs(l_occ).mean(),
        rho_rev = corr(l_rev, -gap),
        rho_local = corr(l_local, -gap),
        rho_norm = corr(l_norm, -gap),
        rho_sep = corr(l_sep, -gap),
        rho_dara = corr(l_dara, -gap),
        rho_occ = corr(l_occ, -gap),
    )

def run(cases = tuple(k for k in CASES if k != 'behavior_m' and not k.startswith('paired')), *, seeds = (0,), **kwargs):
    rows, keys = [], None
    for seed in seeds:
        for name in cases:
            t0 = time.time()
            r = evaluate_case(name, seed = seed, **kwargs)
            if keys is None:
                keys = [k for k in r if k != 'case']
                print('seed  case         ' + ''.join(f'{k:>10}' for k in keys), flush = True)
            print(f"{seed:<6}{r['case']:<13}" + ''.join(f'{r[k]:>10.3f}' for k in keys) + f'   ({time.time() - t0:.0f}s)', flush = True)
            rows.append(dict(seed = seed, **r))
    return rows

if __name__ == '__main__':
    run()
