from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy.stats import norm

# linear gaussian system
# s' = A s + B a + eps,  eps ~ N(0, sigma^2 I)
# a  = K s + eta,        eta ~ N(0, sigma_a^2 I)
# (A, B) is dynamics, K is behavior / occupancy, both controlled independently

Domain = namedtuple('Domain', ['A', 'B', 'K', 'sigma', 'sigma_a'])
Transitions = namedtuple('Transitions', ['state', 'action', 'next_state'])

# helpers

def mat(m):
    return np.atleast_2d(np.asarray(m, dtype = float))

def mean_next_state(domain, state, action):
    return state @ mat(domain.A).T + action @ mat(domain.B).T

def log_normal(x, mu, sigma):
    return norm.logpdf(x, mu, sigma).sum(-1)

# rollouts

def rollout(domain, *, num_traj, horizon, seed = 0, s0_std = 1.):
    rng = np.random.default_rng(seed)
    A, K = mat(domain.A), mat(domain.K)
    s = rng.normal(0., s0_std, (num_traj, A.shape[0]))
    states, actions, next_states = [], [], []

    for _ in range(horizon):
        a = s @ K.T + rng.normal(0., domain.sigma_a, (num_traj, K.shape[0]))
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
