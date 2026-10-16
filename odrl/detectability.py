from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from adaptation.clustering import kmeans, assign
from adaptation.networks import train_domain_classifier
from adaptation.sas_classifier import DynamicsClassifier
from adaptation.joint_cluster_classifier import JointClusterClassifier
from adaptation.normalized_cluster_classifier import ClusterPredictor, train_cluster_predictor, NormalizedClusterClassifier
from adaptation.score import reverse_log_ratio, dara_log_ratio, local_log_ratio, normalized_log_ratio, separate_log_ratio
from odrl.datasets import load_d4rl, split, take
from odrl.evaluate_ratio import CASES, DEVICE, to_tensor, corr, batched, fit_estimators, load_target
from odrl.ground_truth import make_source_env, dynamics_gap

# gate 2, detectability: held-out domain log-loss gain from the next-state channel, in nats per sample
#   coarse  I(D; K | Z)  ≈ CE(occ branch) - CE(joint branch) of the joint classifier   what ours / lodada can use
#   full    I(D; S' | Z) ≈ CE(occ branch) - CE(D_SAS)                                  what dara can use
# cross-fitted over target folds: train on source train + the other folds, score a balanced pair (held-out source, test fold)
# *_shuf = same gain when K (resp. s') is permuted across the pooled data, so I(D; . | Z) = 0 by construction:
#   the capacity / optimization artefact of the bigger model on this very data, the per-row floor
# *_fit = same gain on the training data, the overfitting reference
# null and behavior have I(D; . | Z) = 0 exactly (same physics, deterministic transitions): the finite-sample floor at the same N_t
# gate s, support: experiment 2 / 3 scores restricted to the overlap region of the occupancy propensity e(z) = sigmoid(logit D_SA)
#   omega_tau = { tau <= e(z) <= 1 - tau } on held-out source transitions, frac = its share of the held-out set

TAUS = (0., 0.02, 0.1, 0.2)

# cluster_on: 'next' = kmeans on s' (as experiment 3), 'delta' = kmeans on s' - s, where a contact impulse is visible (gate r, friction)

def feature_of(data, cluster_on):
    return data.next_state if cluster_on == 'next' else data.next_state - data.state

# helpers

def balanced_ce(logits):
    # first half source (0), second half target (1)
    labels = torch.cat((torch.zeros(len(logits) // 2), torch.ones(len(logits) - len(logits) // 2)))
    return F.binary_cross_entropy_with_logits(torch.as_tensor(logits).float(), labels).item()

def pair(src, tar):
    return tuple(torch.cat((s, t)) for s, t in zip(src, tar))

def shuffle_field(data, i, rng):
    return data[:i] + (data[i][torch.as_tensor(rng.permutation(len(data[i])))],) + data[i + 1:]

def pooled_shuffle(src, tar, i, rng):
    # permute field i across both domains: it becomes independent of (z, domain), so I(D; field | Z) = 0 exactly
    n = len(src[i])
    pooled = shuffle_field((torch.cat((src[i], tar[i])),), 0, rng)[0]
    return src[:i] + (pooled[:n],) + src[i + 1:], tar[:i] + (pooled[n:],) + tar[i + 1:]

def gains(joint, sas, norm, data, *, device):
    s, a, ns, c = data
    j = batched(lambda s, a, c: torch.cat(joint(s, a, c), -1), s, a, c, device = device)   # (n, 2): joint, occ branch
    n = batched(lambda s, a, c: torch.cat(norm(s, a, c), -1), s, a, c, device = device)
    d = batched(lambda s, a, ns: sas(s, a, ns).squeeze(-1), s, a, ns, device = device)
    ce_occ = balanced_ce(j[:, 1])
    return ce_occ, ce_occ - balanced_ce(j[:, 0]), ce_occ - balanced_ce(d), balanced_ce(n[:, 1]) - balanced_ce(n[:, 0])

# g_norm = the same coarse gain for the normalized model (its own occ branch as reference)
GAIN_KEYS = ('ce_occ', 'g_coarse', 'g_coarse_shuf', 'g_full', 'g_full_shuf', 'g_norm', 'g_norm_shuf', 'g_coarse_fit', 'g_full_fit')

def cross_fit(train, held, tar, *, num_clusters, steps, seed, device, folds = 5, cluster_on = 'next'):
    gen = torch.Generator().manual_seed(seed)
    n = len(tar.state)
    fold_of = torch.empty(n, dtype = torch.long)
    fold_of[torch.randperm(n, generator = gen)] = torch.arange(n) % folds
    fold_of = fold_of.numpy()
    held_perm = torch.randperm(len(held.state), generator = gen).numpy()
    ds, da = train.state.shape[-1], train.action.shape[-1]
    sac = lambda t: (t[0], t[1], t[3])
    rows = []
    for f in range(folds):
        rng = np.random.default_rng(seed + f)
        tr, te = take(tar, fold_of != f), take(tar, fold_of == f)
        probe = take(held, held_perm[f::folds][:len(te.state)])   # disjoint held-out source per fold, same size as the test fold
        scale = feature_of(tr, cluster_on).std(0) + 1e-6
        centroids = kmeans(feature_of(tr, cluster_on) / scale, num_clusters, seed = seed)
        tensors = lambda data: to_tensor(data, assign(feature_of(data, cluster_on) / scale, centroids))
        src_t, tar_t = tensors(train), tensors(tr)
        src_s, tar_s = pooled_shuffle(*pooled_shuffle(src_t, tar_t, 3, rng), 2, rng)   # shuffled control: same models, no signal

        models = []
        for src_i, tar_i in ((src_t, tar_t), (src_s, tar_s)):
            torch.manual_seed(seed + f)
            joint = train_domain_classifier(JointClusterClassifier(ds, da, num_clusters).to(device), sac(src_i), sac(tar_i), steps = steps)
            torch.manual_seed(seed + f)
            sas = train_domain_classifier(DynamicsClassifier(ds, da).to(device), src_i[:3], tar_i[:3], steps = steps)
            torch.manual_seed(seed + f)
            pred = train_cluster_predictor(ClusterPredictor(ds, da, num_clusters).to(device), sac(src_i), steps = steps)
            norm = train_domain_classifier(NormalizedClusterClassifier(ds, da, num_clusters, pred).to(device), sac(src_i), sac(tar_i), steps = steps)
            models.append((joint, sas, norm))

        test = pair(tensors(probe), tensors(te))
        test_s = shuffle_field(shuffle_field(test, 3, rng), 2, rng)
        fit_src = take(train, rng.choice(len(train.state), len(tr.state), replace = False))
        fit = pair(tensors(fit_src), tar_t)
        g, g_s, g_fit = gains(*models[0], test, device = device), gains(*models[1], test_s, device = device), gains(*models[0], fit, device = device)
        rows.append((g[0], g[1], g_s[1], g[2], g_s[2], g[3], g_s[3], g_fit[1], g_fit[2]))
    return np.mean(rows, 0)

def support_rows(name, scores, gap, replica = None, taus = TAUS):
    e = 1. / (1. + np.exp(-scores['occ']))
    rows = []
    for tau in taus:
        m = (e >= tau) & (e <= 1. - tau)
        r = dict(case = name, tau = tau, frac = m.mean())
        r.update({f'abs_{k}': np.abs(v[m]).mean() for k, v in scores.items()})
        r.update({f'cabs_{k}': np.abs(v[m] - v[m].mean()).mean() for k, v in scores.items()})   # in-band dispersion, a constant offset does not change a ranking
        r.update({f'rho_{k}': corr(v[m], -gap[m]) if m.sum() > 10 else np.nan for k, v in scores.items()})
        if replica is not None:
            # signed mean and the spearman between two independently trained estimators on the same held-out points:
            # a systematic leak is reproducible across fits, estimation noise is not
            r.update({f'mean_{k}': v[m].mean() for k, v in scores.items()})
            r.update({f'rep_{k}': corr(v[m], replica[k][m]) if m.sum() > 10 else np.nan for k, v in scores.items()})
        rows.append(r)
    return rows

def evaluate_case(name, *, prefix = 'hopper', axis = 'gravity', num_clusters = 10, steps = 3000, num_target = 5000, num_held = 50_000, seed = 0, device = DEVICE, gate_p = True, replicate = False, cluster_on = 'next'):
    train, held, null = split(load_d4rl(f'{prefix}-medium-replay-v2'), num_held = num_held, num_null = num_target, seed = seed)
    tar, tar_env = load_target(name, null, num_target = num_target, seed = seed, prefix = prefix, axis = axis)
    src_env = make_source_env(f'{prefix}-{axis}')

    gain = cross_fit(train, held, tar, num_clusters = num_clusters, steps = steps, seed = seed, device = device, cluster_on = cluster_on) if gate_p else np.full(len(GAIN_KEYS), np.nan)

    # experiment 2 / 3 estimators on the full target, as odrl.evaluate_ratio
    scale = tar.next_state.std(0) + 1e-6   # gap normalization stays on s'
    cscale = feature_of(tar, cluster_on).std(0) + 1e-6
    centroids = kmeans(feature_of(tar, cluster_on) / cscale, num_clusters, seed = seed)
    cluster_of = lambda data: assign(feature_of(data, cluster_on) / cscale, centroids)
    src_t, tar_t = to_tensor(train, cluster_of(train)), to_tensor(tar, cluster_of(tar))
    s, a, ns, c = to_tensor(held, cluster_of(held))
    def scores_of(fit_seed):
        occ, sas, joint, rev, norm, zk = fit_estimators(src_t, tar_t, num_clusters = num_clusters, steps = steps, seed = fit_seed, device = device)
        return dict(
            rev = batched(lambda s, a, c: reverse_log_ratio(rev, s, a, c), s, a, c, device = device),
            dara = batched(lambda s, a, ns: dara_log_ratio(sas, occ, s, a, ns), s, a, ns, device = device),
            local = batched(lambda s, a, c: local_log_ratio(joint, s, a, c), s, a, c, device = device),
            norm = batched(lambda s, a, c: normalized_log_ratio(norm, s, a, c), s, a, c, device = device),
            sep = batched(lambda s, a, c: separate_log_ratio(zk, occ, s, a, c), s, a, c, device = device),
            occ = batched(lambda s, a: occ(s, a).squeeze(-1), s, a, device = device),
        )

    scores = scores_of(seed)
    replica = scores_of(seed + 1000) if replicate else None   # same data, same clusters, independent fit
    gap, _ = dynamics_gap(src_env, tar_env if tar_env is not None else src_env, held.state, held.action, scale = scale)
    return dict(zip(GAIN_KEYS, gain)), support_rows(name, scores, gap, replica)

def run(cases = tuple(k for k in CASES if k != 'behavior_m' and not k.startswith('paired')), *, seeds = (0,), **kwargs):
    gain_rows, sup_rows = [], []
    for seed in seeds:
        for name in cases:
            t0 = time.time()
            gain, sup = evaluate_case(name, seed = seed, **kwargs)
            gain_rows.append(dict(seed = seed, case = name, **gain))
            sup_rows.extend(dict(seed = seed, **r) for r in sup)
            print(f'gain  {seed:<4}{name:<13}' + ''.join(f'{k} {v:>8.4f}  ' for k, v in gain.items()) + f'({time.time() - t0:.0f}s)', flush = True)
            for r in sup:
                print(f'supp  {seed:<4}{name:<13}tau {r["tau"]:<5}frac {r["frac"]:.3f}  ' + ''.join(f'{k} {r[k]:>8.3f}  ' for k in r if k.startswith(('abs_', 'cabs_', 'rho_', 'mean_', 'rep_'))), flush = True)
    return gain_rows, sup_rows

if __name__ == '__main__':
    run()
