from __future__ import annotations

import argparse
import numpy as np
import h5py

from locomotion.truth import HopperSim

# smoke-test data only: hopper rollouts under a noisy periodic controller, written in the odrl hdf5 layout
# this is NOT the d4rl / odrl benchmark data; it exists so the pipeline can be exercised before those files are available

def collect(path, *, shift, level, num_transitions, action_noise, seed, max_steps = 200):
    rng = np.random.default_rng(seed)
    sim = HopperSim(shift, level)
    env = sim.env
    obs_, act_, term_, timeout_ = [], [], [], []
    while len(obs_) < num_transitions:
        obs, _ = env.reset(seed = int(rng.integers(1 << 31)))
        phase, amp = rng.uniform(0, 2 * np.pi, 3), rng.uniform(0.2, 0.8)
        for t in range(max_steps):
            action = np.clip(amp * np.sin(0.3 * t + phase) + action_noise * rng.normal(size = 3), -1, 1)
            next_obs, _, terminated, _, _ = env.step(action)
            obs_.append(obs); act_.append(action)
            timeout = t == max_steps - 1
            term_.append(terminated); timeout_.append(timeout and not terminated)
            obs = next_obs
            if terminated or timeout:
                break
    with h5py.File(path, 'w') as f:
        f['observations'] = np.array(obs_, dtype = np.float32)
        f['actions'] = np.array(act_, dtype = np.float32)
        f['rewards'] = np.zeros(len(obs_), dtype = np.float32)
        f['terminals'] = np.array(term_)
        f['timeouts'] = np.array(timeout_)
    print(f'{path}: {len(obs_)} transitions, shift {shift} x{level}, action noise {action_noise}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required = True)
    ap.add_argument('--shift', default = None, choices = [None, 'gravity', 'friction'])
    ap.add_argument('--level', type = float, default = 1.)
    ap.add_argument('--num', type = int, default = 50_000)
    ap.add_argument('--noise', type = float, default = 0.5)
    ap.add_argument('--seed', type = int, default = 0)
    a = ap.parse_args()
    collect(a.out, shift = a.shift, level = a.level, num_transitions = a.num, action_noise = a.noise, seed = a.seed)
