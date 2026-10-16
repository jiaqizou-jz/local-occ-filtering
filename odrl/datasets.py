from __future__ import annotations

from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

Transitions = namedtuple('Transitions', ['state', 'action', 'next_state'])

DATA_DIR = Path(__file__).resolve().parent.parent / 'dataset' / 'mujoco'

# helpers

def as_f32(*xs):
    return tuple(np.asarray(x, dtype = np.float32) for x in xs)

def take(data, idx):
    return Transitions(*(x[idx] for x in data))

# source: d4rl, target: odrl hdf5 (dataset/mujoco/download.sh) or another d4rl set for the same-physics control

@lru_cache(maxsize = None)
def load_d4rl(name):
    import gym, d4rl
    d = d4rl.qlearning_dataset(gym.make(name))
    return Transitions(*as_f32(d['observations'], d['actions'], d['next_observations']))

@lru_cache(maxsize = None)
def load_odrl(env_name, shift_level, quality):
    path = DATA_DIR / f"{env_name.replace('-', '_')}_{shift_level}_{quality}.hdf5"
    with h5py.File(path, 'r') as h:
        return Transitions(*as_f32(h['observations'][:], h['actions'][:], h['next_observations'][:]))

def split(data, *, num_held, num_null, seed = 0):
    # disjoint random chunks of one dataset: train (rest), held-out probes, null target
    perm = np.random.default_rng(seed).permutation(len(data.state))
    held, null, train = perm[:num_held], perm[num_held:num_held + num_null], perm[num_held + num_null:]
    return take(data, train), take(data, held), take(data, null)

def subsample(data, num, *, seed = 0):
    return take(data, np.random.default_rng(seed).choice(len(data.state), num, replace = False))
