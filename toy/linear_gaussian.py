from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp

# linear gaussian system
# s' = A s + B a + c + eps,  eps ~ N(0, sigma^2 I)
# a  = K s + b + eta,        eta ~ N(0, sigma_a^2 I)
# (A, B, c) is dynamics, (K, b) is behavior / occupancy, both controlled independently; offsets default to 0

Domain = namedtuple('Domain', ['A', 'B', 'K', 'sigma', 'sigma_a', 'c', 'b'], defaults = (0., 0.))
Transitions = namedtuple('Transitions', ['state', 'action', 'next_state'])

# helpers

def mat(m):
    return np.atleast_2d(np.asarray(m, dtype = float))

def mean_next_state(domain, state, action):
    return state @ mat(domain.A).T + action @ mat(domain.B).T + domain.c

def log_normal(x, mu, sigma):
    return norm.logpdf(x, mu, sigma).sum(-1)

# rollouts

def rollout(domain, *, num_traj, horizon, seed = 0, s0_std = 1., s0_mean = 0.):
    rng = np.random.default_rng(seed)
    A, K = mat(domain.A), mat(domain.K)
    s = rng.normal(s0_mean, s0_std, (num_traj, A.shape[0]))
    states, actions, next_states = [], [], []

    for _ in range(horizon):
        a = s @ K.T + domain.b + rng.normal(0., domain.sigma_a, (num_traj, K.shape[0]))
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

# joint of (s_t, a_t, s_{t+1}) is gaussian for every t and a rollout dataset is a uniform mixture over t, 1d only
# mean: m_{t+1} = g m_t + B b + c, var: v_{t+1} = g^2 v_t + B^2 sigma_a^2 + sigma^2, g = A + B K, s_0 ~ N(s0_mean, s0_std^2)

def joint_moments(domain, *, horizon, s0_std = 1., s0_mean = 0.):
    A, B, K = (float(np.asarray(x)) for x in (domain.A, domain.B, domain.K))
    g, va, vn = A + B * K, domain.sigma_a ** 2, domain.sigma ** 2
    means, covs, m, v = [], [], s0_mean, s0_std ** 2

    for _ in range(horizon):
        cas = A * K * v + B * (K * K * v + va)
        means.append([m, K * m + domain.b, g * m + B * domain.b + domain.c])
        covs.append([[v, K * v, g * v], [K * v, K * K * v + va, cas], [g * v, cas, g * g * v + B * B * va + vn]])
        m, v = g * m + B * domain.b + domain.c, g * g * v + B * B * va + vn

    return np.array(means), np.array(covs)

def mixture_logpdf(x, means, covs):
    logp = np.stack([multivariate_normal(m, c).logpdf(x) for m, c in zip(means, covs)])
    return logsumexp(logp, axis = 0) - np.log(len(covs))

# exact lodada target at the s' level: log p_t(s, a | s') - log p_s(s, a | s'), s0_mean = (source, target) initial state means

def true_reverse_log_ratio(src, tar, state, action, next_state, *, horizon, s0_std = 1., s0_mean = (0., 0.)):
    x = np.concatenate((state, action, next_state), -1)
    out = 0.
    for domain, sign, mean in ((tar, 1., s0_mean[1]), (src, -1., s0_mean[0])):
        means, covs = joint_moments(domain, horizon = horizon, s0_std = s0_std, s0_mean = mean)
        out = out + sign * (mixture_logpdf(x, means, covs) - mixture_logpdf(next_state, means[:, 2:], covs[:, 2:, 2:]))
    return out

# stationary (s, a) moments of the closed loop, 1d: var(s) = (B^2 sigma_a^2 + sigma^2) / (1 - g^2)

def stationary_std(domain):
    A, B, K = (float(np.asarray(x)) for x in (domain.A, domain.B, domain.K))
    g = A + B * K
    return np.sqrt((B * B * domain.sigma_a ** 2 + domain.sigma ** 2) / (1. - g * g))

def stationary_sa_cov(domain):
    K, v = float(np.asarray(domain.K)), stationary_std(domain) ** 2
    return np.array([[v, K * v], [K * v, K * K * v + domain.sigma_a ** 2]])
