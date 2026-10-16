from __future__ import annotations

import numpy as np

from toy.linear_gaussian import rollout, true_reverse_log_ratio, stationary_std, stationary_sa_cov
from toy.generate_dataset import SOURCE, CASES
from toy.evaluate_ratio import evaluate, print_rows

# experiment 1b: adversarial false negative
# l_rev = l_dyn + l_occ - l_marg vanishes iff the target occupancy is the exponential tilt p_t(s, a) ∝ p_s(s, a) exp(-h(s, a))
# where l_dyn(s, a, s') = h(s, a) + m(s'): gaussian dynamics give l_dyn ∝ (mu_t - mu_s)(2 s' - mu_s - mu_t), separable only
# when mu_t - mu_s is a constant c (offset shift, gravity-like), never for a gain shift (A, B)
# for the offset shift the tilt is a mean shift delta = (c / sigma^2) Sigma_sa (A, B)^T of the stationary source occupancy,
# realized as one-step data: s ~ N(delta_s, v), a = K s + b + eta with b = delta_a - K delta_s, s' under target dynamics
# cases (one-step stationary sampling): A same, C gain shift, F offset shift with source occupancy, E offset shift with the tilt
# pass, over all seeds: abs_rev(E) <= 2 x abs_rev(A) while abs_local(E) >= 0.5 x abs_local(F) and rho_local(E) > 0.9
# run with: python -m toy.adversarial

OFFSET = 0.1 * np.sqrt(2 * 1.41)   # kl of the offset shift matches cases c / d: c^2 / (2 sigma^2) = 1.41
STD = stationary_std(SOURCE)

OFFSET_SHIFT = SOURCE._replace(c = OFFSET)

def tilt(domain):
    A, B, K = (float(np.asarray(x)) for x in (domain.A, domain.B, domain.K))
    delta_s, delta_a = stationary_sa_cov(domain) @ np.array([A, B]) * domain.c / domain.sigma ** 2
    return delta_s, delta_a - K * delta_s

DELTA_S, B_TILT = tilt(OFFSET_SHIFT)

# (domain, initial state mean) per case

ONE_STEP = dict(
    A = (SOURCE, 0.),
    C = (CASES['C'], 0.),
    F = (OFFSET_SHIFT, 0.),
    E = (OFFSET_SHIFT._replace(b = B_TILT), DELTA_S),
)

def one_step(domain, *, num, seed, s_mean = 0.):
    return rollout(domain, num_traj = num, horizon = 1, seed = seed, s0_std = STD, s0_mean = s_mean)

def exact_abs_rev(name, held):
    tar, s_mean = ONE_STEP[name]
    return np.abs(true_reverse_log_ratio(SOURCE, tar, held.state, held.action, held.next_state, horizon = 1, s0_std = STD, s0_mean = (0., s_mean))).mean()

def evaluate_one_step(name, *, num_source = 50_000, num_target = 5_000, seed = 0, **kwargs):
    tar_domain, s_mean = ONE_STEP[name]
    src = one_step(SOURCE, num = num_source, seed = seed)
    tar = one_step(tar_domain, num = num_target, seed = seed + 1, s_mean = s_mean)
    held = one_step(SOURCE, num = num_source, seed = seed + 100)
    return evaluate(name, src, tar, held, SOURCE, tar_domain, seed = seed, horizon = 1, s0_std = STD, s0_mean = (0., s_mean), **kwargs)

def run(seeds = (0, 1, 2)):
    held = one_step(SOURCE, num = 50_000, seed = 100)
    print(f'stationary std {STD:.4f}, offset c {OFFSET:.3f}, tilt delta_s {DELTA_S:+.3f}, behavior offset b {B_TILT:+.3f}')
    print('exact mean |l_rev| on held-out source: ' + '  '.join(f'{k} {exact_abs_rev(k, held):.3f}' for k in ONE_STEP) + '\n')

    rows = [evaluate_one_step(name, seed = seed) for seed in seeds for name in ONE_STEP]
    print_rows(rows)
    return rows

if __name__ == '__main__':
    run()

# cluster-level exact reverse ratio: l_rev_C = l_dyn_C + l_occ - l_marg_C, all closed form for one-step gaussian data
# the s'-level cancellation is pointwise in s'; after binning s' the bounded coarse dynamics term can no longer cancel the occupancy tilt

def exact_cluster_reverse(name, held, edges):
    from scipy.stats import norm, multivariate_normal
    from toy.linear_gaussian import joint_moments, true_cluster_log_ratio
    from adaptation.clustering import assign_bins

    tar, s_mean = ONE_STEP[name]
    c = assign_bins(held.next_state, edges)
    l_dyn = true_cluster_log_ratio(SOURCE, tar, held.state, held.action, c, edges)

    z = np.concatenate((held.state, held.action), -1)
    full_edges = np.concatenate(([-np.inf], edges, [np.inf]))
    l_occ, l_marg = 0., 0.
    for domain, sign, mean in ((tar, 1., s_mean), (SOURCE, -1., 0.)):
        means, covs = joint_moments(domain, horizon = 1, s0_std = STD, s0_mean = mean)
        l_occ = l_occ + sign * multivariate_normal(means[0, :2], covs[0, :2, :2]).logpdf(z)
        m, sd = means[0, 2], np.sqrt(covs[0, 2, 2])
        p_bin = norm.cdf(full_edges[1:], m, sd) - norm.cdf(full_edges[:-1], m, sd)
        l_marg = l_marg + sign * np.log(p_bin[c] + 1e-12)

    return l_dyn + l_occ - l_marg, l_dyn
