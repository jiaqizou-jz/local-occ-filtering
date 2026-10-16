from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from adaptation.clustering import kmeans, assign
from adaptation.networks import train_domain_classifier
from adaptation.sas_classifier import DynamicsClassifier
from adaptation.joint_cluster_classifier import JointClusterClassifier
from adaptation.score import reverse_log_ratio, dara_log_ratio, local_log_ratio
from odrl.datasets import load_d4rl, split, take
from odrl.evaluate_ratio import CASES, SOURCE, ENV, DEVICE, to_tensor, corr, batched, fit_estimators, load_target
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

def gains(joint, sas, data, *, device):
    s, a, ns, c = data
    j = batched(lambda s, a, c: torch.cat(joint(s, a, c), -1), s, a, c, device = device)   # (n, 2): joint, occ branch
    d = batched(lambda s, a, ns: sas(s, a, ns).squeeze(-1), s, a, ns, device = device)
    ce_occ = balanced_ce(j[:, 1])
    return ce_occ, ce_occ - balanced_ce(j[:, 0]), ce_occ - balanced_ce(d)

GAIN_KEYS = ('ce_occ', 'g_coarse', 'g_coarse_shuf', 'g_full', 'g_full_shuf', 'g_coarse_fit', 'g_full_fit')

def cross_fit(train, held, tar, *, num_clusters, steps, seed, device, folds = 5):
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
        scale = tr.next_state.std(0) + 1e-6
        centroids = kmeans(tr.next_state / scale, num_clusters, seed = seed)
        tensors = lambda data: to_tensor(data, assign(data.next_state / scale, centroids))
        src_t, tar_t = tensors(train), tensors(tr)
        src_s, tar_s = pooled_shuffle(*pooled_shuffle(src_t, tar_t, 3, rng), 2, rng)   # shuffled control: same models, no signal

        models = []
        for src_i, tar_i in ((src_t, tar_t), (src_s, tar_s)):
            torch.manual_seed(seed + f)
            joint = train_domain_classifier(JointClusterClassifier(ds, da, num_clusters).to(device), sac(src_i), sac(tar_i), steps = steps)
            torch.manual_seed(seed + f)
            sas = train_domain_classifier(DynamicsClassifier(ds, da).to(device), src_i[:3], tar_i[:3], steps = steps)
            models.append((joint, sas))

        test = pair(tensors(probe), tensors(te))
        test_s = shuffle_field(shuffle_field(test, 3, rng), 2, rng)
        fit_src = take(train, rng.choice(len(train.state), len(tr.state), replace = False))
        fit = pair(tensors(fit_src), tar_t)
        g, g_s, g_fit = gains(*models[0], test, device = device), gains(*models[1], test_s, device = device), gains(*models[0], fit, device = device)
        rows.append((g[0], g[1], g_s[1], g[2], g_s[2], g_fit[1], g_fit[2]))
    return np.mean(rows, 0)

def support_rows(name, scores, gap, taus = TAUS):
    e = 1. / (1. + np.exp(-scores['occ']))
    rows = []
    for tau in taus:
        m = (e >= tau) & (e <= 1. - tau)
        r = dict(case = name, tau = tau, frac = m.mean())
        r.update({f'abs_{k}': np.abs(v[m]).mean() for k, v in scores.items()})
        r.update({f'rho_{k}': corr(v[m], -gap[m]) if m.sum() > 10 else np.nan for k, v in scores.items()})
        rows.append(r)
    return rows

def evaluate_case(name, *, source = SOURCE, num_clusters = 10, steps = 3000, num_target = 5000, num_held = 50_000, seed = 0, device = DEVICE):
    train, held, null = split(load_d4rl(source), num_held = num_held, num_null = num_target, seed = seed)
    tar, tar_env = load_target(name, null, num_target = num_target, seed = seed)
    src_env = make_source_env(ENV)

    gain = cross_fit(train, held, tar, num_clusters = num_clusters, steps = steps, seed = seed, device = device)

    # experiment 2 / 3 estimators on the full target, as odrl.evaluate_ratio
    scale = tar.next_state.std(0) + 1e-6
    centroids = kmeans(tar.next_state / scale, num_clusters, seed = seed)
    cluster_of = lambda x: assign(x / scale, centroids)
    src_t, tar_t = to_tensor(train, cluster_of(train.next_state)), to_tensor(tar, cluster_of(tar.next_state))
    s, a, ns, c = to_tensor(held, cluster_of(held.next_state))
    occ, sas, joint, rev = fit_estimators(src_t, tar_t, num_clusters = num_clusters, steps = steps, seed = seed, device = device)
    scores = dict(
        rev = batched(lambda s, a, c: reverse_log_ratio(rev, s, a, c), s, a, c, device = device),
        dara = batched(lambda s, a, ns: dara_log_ratio(sas, occ, s, a, ns), s, a, ns, device = device),
        local = batched(lambda s, a, c: local_log_ratio(joint, s, a, c), s, a, c, device = device),
        occ = batched(lambda s, a: occ(s, a).squeeze(-1), s, a, device = device),
    )
    gap, _ = dynamics_gap(src_env, tar_env if tar_env is not None else src_env, held.state, held.action, scale = scale)
    return dict(zip(GAIN_KEYS, gain)), support_rows(name, scores, gap)

def run(cases = tuple(CASES), *, seeds = (0,), **kwargs):
    gain_rows, sup_rows = [], []
    for seed in seeds:
        for name in cases:
            t0 = time.time()
            gain, sup = evaluate_case(name, seed = seed, **kwargs)
            gain_rows.append(dict(seed = seed, case = name, **gain))
            sup_rows.extend(dict(seed = seed, **r) for r in sup)
            print(f'gain  {seed:<4}{name:<13}' + ''.join(f'{k} {v:>8.4f}  ' for k, v in gain.items()) + f'({time.time() - t0:.0f}s)', flush = True)
            for r in sup:
                print(f'supp  {seed:<4}{name:<13}tau {r["tau"]:<5}frac {r["frac"]:.3f}  ' + ''.join(f'{k} {r[k]:>8.3f}  ' for k in r if k.startswith(('abs_', 'rho_'))), flush = True)
    return gain_rows, sup_rows

if __name__ == '__main__':
    run()
