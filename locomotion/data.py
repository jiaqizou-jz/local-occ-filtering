from __future__ import annotations

from collections import namedtuple

import numpy as np
import h5py

# transitions from d4rl / odrl hdf5 files, keeping the raw observations for the simulator probes
# d4rl v2 files carry next_observations; odrl target files are reconstructed from observations[i + 1] as in dataset/call_dataset.py,
# except that the last step of every episode (terminal or timeout) is dropped instead of being paired with the next episode's first state

Transitions = namedtuple('Transitions', ['state', 'action', 'next_state', 'episode'])

def load_transitions(path, *, max_episode_steps = 1000):
    with h5py.File(path, 'r') as f:
        obs, act, term = f['observations'][:], f['actions'][:], f['terminals'][:].astype(bool)
        timeout = f['timeouts'][:].astype(bool) if 'timeouts' in f else None
        next_obs = f['next_observations'][:] if 'next_observations' in f else None

    if timeout is None:
        step = np.zeros(len(obs), dtype = int)
        for i in range(1, len(obs)):
            step[i] = 0 if term[i - 1] else step[i - 1] + 1
        timeout = step == max_episode_steps - 1
    end = term | timeout
    episode = np.concatenate(([0], np.cumsum(end)[:-1]))

    if next_obs is None:
        keep = ~end[:-1]
        obs, act, next_obs, episode = obs[:-1][keep], act[:-1][keep], obs[1:][keep], episode[:-1][keep]

    f32 = lambda x: np.asarray(x, dtype = np.float32)
    return Transitions(f32(obs), f32(act), f32(next_obs), episode)

def split_episodes(data, *, held_fraction = 0.2, seed = 0):
    rng = np.random.default_rng(seed)
    episodes = np.unique(data.episode)
    held = rng.choice(episodes, int(round(held_fraction * len(episodes))), replace = False)
    mask = np.isin(data.episode, held)
    return subset(data, ~mask), subset(data, mask)

def subset(data, mask):
    return Transitions(*(x[mask] for x in data))

def subsample(data, num, *, seed = 0):
    if num is None or num >= len(data.state):
        return data
    idx = np.random.default_rng(seed).choice(len(data.state), num, replace = False)
    return subset(data, idx)

# standardization with source statistics, shared by both domains

class Standardizer:
    def __init__(self, data):
        self.mean = [x.mean(0) for x in (data.state, data.action, data.next_state)]
        self.std = [x.std(0) + 1e-6 for x in (data.state, data.action, data.next_state)]

    def __call__(self, data):
        return tuple((x - m) / s for x, m, s in zip((data.state, data.action, data.next_state), self.mean, self.std))
