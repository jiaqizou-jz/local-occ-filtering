from __future__ import annotations

import numpy as np

# shared deterministic mapping C = g(s')
# fit on target next states only, then assign both domains with the same centroids / edges

def assign(x, centroids):
    return ((x[:, None, :] - centroids[None]) ** 2).sum(-1).argmin(-1)

def kmeans(x, num_clusters, *, iters = 50, seed = 0):
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(len(x), num_clusters, replace = False)]

    for _ in range(iters):
        c = assign(x, centroids)
        centroids = np.stack([x[c == k].mean(0) if (c == k).any() else centroids[k] for k in range(num_clusters)])

    return centroids

# 1d quantile bins, the toy-system stand-in for k-means

def quantile_edges(x, num_clusters):
    return np.quantile(x.reshape(-1), np.linspace(0., 1., num_clusters + 1)[1:-1])

def assign_bins(x, edges):
    return np.searchsorted(edges, x.reshape(-1))
