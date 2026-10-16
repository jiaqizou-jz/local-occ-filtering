from __future__ import annotations

import time

import numpy as np
import gym, d4rl

from adaptation.clustering import kmeans, assign
from odrl.datasets import load_odrl
from odrl.ground_truth import make_source_env, make_target_env

# resolution test for the coarse channel K = g(outcome), no classifier involved
# mujoco is deterministic: given z the cluster label is a point mass in each domain, so the coarse channel carries at most
#   I(D; K | Z) <= ln 2 * P_z[ g(f_s(z)) != g(f_t(z)) ]
# flip = that probability on held-out source windows, the ceiling for any cluster-level estimator (lodada, ours)
# outcomes after h open-loop steps under the data's own action sequence (same actions in both physics, so the policy is not a confounder):
#   last = s_{t+h}, delta = s_{t+h} - s_t, window = (s_{t+1}, ..., s_{t+h})
# partition: kmeans on target one-step next states (as experiment 3, h = 1 only) or on the pooled simulator outcomes of the
#   probes themselves (favorable: the partition is fit on exactly the paired outcomes it has to separate)
# gap = mean ||f_t - f_s|| / std of the outcome, the paired effect size (not a kl: both conditional laws are point masses)
# recon = median ||f_s - data|| / std after h open-loop source-physics steps, checks the simulator replays the data

SOURCE = 'hopper-medium-replay-v2'
ENV = 'hopper-gravity'
SHIFT_LEVELS = (0.5, 2.0)
HORIZONS = (1, 5, 20)
NUM_CLUSTERS = (10, 30, 100, 300, 1000)
FEATURES = ('last', 'delta', 'window')

# helpers

def windows(name, horizon, *, num, seed):
    d = gym.make(name).get_dataset()
    obs, act, nobs, done = d['observations'], d['actions'], d['next_observations'], d['terminals'] | d['timeouts']
    # start i is valid when the episode runs through i + horizon - 1: no end flag at i .. i + horizon - 2
    ok = np.ones(len(obs) - horizon + 1, dtype = bool)
    for j in range(horizon - 1):
        ok &= ~done[j:j + len(ok)]
    starts = np.sort(np.random.default_rng(seed).choice(np.where(ok)[0], num, replace = False))
    idx = starts[:, None] + np.arange(horizon)
    return obs[starts].astype(np.float64), act[idx].astype(np.float64), nobs[idx].astype(np.float64)

def rollout(env, states, action_seqs):
    sim = env.unwrapped
    nq = sim.model.nq
    env.reset()
    out = np.empty(action_seqs.shape[:2] + (states.shape[-1],))
    for i, (s, acts) in enumerate(zip(states, action_seqs)):
        sim.set_state(np.concatenate(([0.], s[:nq - 1])), s[nq - 1:])
        for j, a in enumerate(acts):
            out[i, j] = sim.step(a)[0]
    return out

def outcome(traj, s0, feature):
    if feature == 'last':
        return traj[:, -1]
    if feature == 'delta':
        return traj[:, -1] - s0
    return traj.reshape(len(traj), -1)

def flip_rate(fit, f_src, f_tar, num_clusters, *, seed):
    scale = fit.std(0) + 1e-6
    centroids = kmeans(fit / scale, num_clusters, seed = seed)
    return (assign(f_src / scale, centroids) != assign(f_tar / scale, centroids)).mean()

def run(*, shift_levels = SHIFT_LEVELS, horizons = HORIZONS, num_clusters = NUM_CLUSTERS, features = FEATURES, num = 20_000, seed = 0):
    h_max = max(horizons)
    s0, acts, data = windows(SOURCE, h_max, num = num, seed = seed)
    t0 = time.time()
    src_env = make_source_env(ENV)
    traj = {'src': rollout(src_env, s0, acts)}
    traj.update({g: rollout(make_target_env(ENV, g), s0, acts) for g in shift_levels})
    print(f'{num} windows of {h_max} steps, {1 + len(shift_levels)} physics rolled out in {time.time() - t0:.0f}s', flush = True)

    print('gravity  h   feature  partition         gap     recon' + ''.join(f'  flip@K={k:<5}' for k in num_clusters), flush = True)
    for g in shift_levels:
        for h in horizons:
            for feature in features:
                if feature == 'window' and h == 1:
                    continue
                f_src, f_tar = (outcome(traj[key][:, :h], s0, feature) for key in ('src', g))
                f_data = outcome(data[:, :h], s0, feature)
                scale = f_src.std(0) + 1e-6
                gap = (np.linalg.norm((f_tar - f_src) / scale, axis = -1)).mean()
                recon = np.median(np.linalg.norm((f_src - f_data) / scale, axis = -1))

                partitions = dict(pooled = np.concatenate((f_src, f_tar)))
                if h == 1 and feature != 'delta':
                    partitions['target'] = load_odrl(ENV, g, 'medium').next_state.astype(np.float64)
                for name, fit in partitions.items():
                    flips = [flip_rate(fit, f_src, f_tar, k, seed = seed) for k in num_clusters]
                    print(f'{g:<8} {h:<3} {feature:<8} {name:<10}{gap:>10.3f}{recon:>10.3f}' + ''.join(f'{f:>13.4f}' for f in flips), flush = True)

if __name__ == '__main__':
    run()
