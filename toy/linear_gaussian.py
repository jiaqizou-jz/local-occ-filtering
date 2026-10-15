from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy.stats import norm

# linear gaussian system
# s' = A s + B a + d + eps,  eps ~ N(0, sigma^2 I)
# a  = K s + b + eta,        eta ~ N(0, sigma_a^2 I)
# (A, B, d) is dynamics, (K, b, sigma_a) is behavior / occupancy, both controlled independently
# d (drift) and b (action offset) default to 0; a drift shift is the separable dynamics shift used by the adversarial case

Domain = namedtuple('Domain', ['A', 'B', 'K', 'sigma', 'sigma_a', 'd', 'b'], defaults = (0., 0.))
Transitions = namedtuple('Transitions', ['state', 'action', 'next_state'])

# helpers

def mat(m):
    return np.atleast_2d(np.asarray(m, dtype = float))

def mean_next_state(domain, state, action):
    return state @ mat(domain.A).T + action @ mat(domain.B).T + domain.d

def mean_action(domain, state):
    return state @ mat(domain.K).T + domain.b

def log_normal(x, mu, sigma):
    return norm.logpdf(x, mu, sigma).sum(-1)

# rollouts

def rollout(domain, *, num_traj, horizon, seed = 0, s0_std = 1.):
    rng = np.random.default_rng(seed)
    A, K = mat(domain.A), mat(domain.K)
    s = rng.normal(0., s0_std, (num_traj, A.shape[0]))
    states, actions, next_states = [], [], []

    for _ in range(horizon):
        a = mean_action(domain, s) + rng.normal(0., domain.sigma_a, (num_traj, K.shape[0]))
        ns = mean_next_state(domain, s, a) + rng.normal(0., domain.sigma, s.shape)
        states.append(s), actions.append(a), next_states.append(ns)
        s = ns

    return Transitions(*(np.concatenate(x) for x in (states, actions, next_states)))

# ground truth
# equal-variance case reduces to (mu_s - mu_t)^2 / (2 sigma^2)

def true_dynamics_kl(src, tar, state, action):
    mu_s, mu_t = mean_next_state(src, state, action), mean_next_state(tar, state, action)
    var_s, var_t = src.sigma ** 2, tar.sigma ** 2
    return (np.log(tar.sigma / src.sigma) + (var_s + (mu_s - mu_t) ** 2) / (2 * var_t) - 0.5).sum(-1)

def true_next_state_log_ratio(src, tar, state, action, next_state):
    return log_normal(next_state, mean_next_state(tar, state, action), tar.sigma) - log_normal(next_state, mean_next_state(src, state, action), src.sigma)

# 1d only: P(C = k | s, a) = Phi((e_k - mu) / sigma) - Phi((e_{k-1} - mu) / sigma)

def log_cluster_prob(domain, state, action, cluster, edges):
    edges = np.concatenate(([-np.inf], edges, [np.inf]))
    mu = mean_next_state(domain, state, action).reshape(-1)
    lo, hi = edges[cluster], edges[cluster + 1]
    return np.log(norm.cdf(hi, mu, domain.sigma) - norm.cdf(lo, mu, domain.sigma) + 1e-12)

def true_cluster_log_ratio(src, tar, state, action, cluster, edges):
    return log_cluster_prob(tar, state, action, cluster, edges) - log_cluster_prob(src, state, action, cluster, edges)

# exact rollout occupancy (1d, s0 ~ N(0, s0_std^2))
# s_h is gaussian with mean m_h and variance v_h: m_{h+1} = G m_h + B b + d, v_{h+1} = G^2 v_h + B^2 sigma_a^2 + sigma^2, G = A + B K
# a dataset of horizon H mixes h = 0..H-1 uniformly, so p(s) and p(s') are gaussian mixtures with known moments

def closed_loop_gain(domain):
    A, B, K = (float(mat(x)[0, 0]) for x in (domain.A, domain.B, domain.K))
    return A + B * K

def state_moments(domain, horizon, s0_std = 1.):
    G, B = closed_loop_gain(domain), float(mat(domain.B)[0, 0])
    noise = B ** 2 * domain.sigma_a ** 2 + domain.sigma ** 2
    m, v = [0.], [s0_std ** 2]
    for _ in range(horizon):
        m.append(G * m[-1] + B * domain.b + domain.d)
        v.append(G ** 2 * v[-1] + noise)
    return np.array(m), np.array(v)

def state_variances(domain, horizon, s0_std = 1.):
    return state_moments(domain, horizon, s0_std)[1]

def log_mixture_normal(x, means, variances):
    x = np.asarray(x).reshape(-1)
    logp = norm.logpdf(x[:, None], means[None], np.sqrt(variances)[None])
    return np.logaddexp.reduce(logp, axis = 1) - np.log(len(variances))

def log_occupancy(domain, state, action, horizon, s0_std = 1.):
    m, v = state_moments(domain, horizon, s0_std)
    return log_mixture_normal(state, m[:-1], v[:-1]) + norm.logpdf(action.reshape(-1), mean_action(domain, state).reshape(-1), domain.sigma_a)

def log_cluster_marginal(domain, cluster, edges, horizon, s0_std = 1.):
    m, v = state_moments(domain, horizon, s0_std)
    edges = np.concatenate(([-np.inf], edges, [np.inf]))
    mu, sd = m[1:][None], np.sqrt(v[1:])[None]
    prob = (norm.cdf(edges[1:, None], mu, sd) - norm.cdf(edges[:-1, None], mu, sd)).mean(1)
    return np.log(prob[cluster] + 1e-12)

def true_occupancy_log_ratio(src, tar, state, action, horizon, s0_std = 1.):
    return log_occupancy(tar, state, action, horizon, s0_std) - log_occupancy(src, state, action, horizon, s0_std)

def true_marginal_log_ratio(src, tar, cluster, edges, horizon, s0_std = 1.):
    return log_cluster_marginal(tar, cluster, edges, horizon, s0_std) - log_cluster_marginal(src, cluster, edges, horizon, s0_std)

# the quantity lodada's reverse classifier estimates: log p_t(s, a | C) / p_s(s, a | C) = dyn + occ - marginal

def true_reverse_log_ratio(src, tar, state, action, cluster, edges, horizon, s0_std = 1.):
    return (true_cluster_log_ratio(src, tar, state, action, cluster, edges)
            + true_occupancy_log_ratio(src, tar, state, action, horizon, s0_std)
            - true_marginal_log_ratio(src, tar, cluster, edges, horizon, s0_std))
