from __future__ import annotations

import sys, os
from pathlib import Path
import numpy as np, torch, gym, yaml

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'algo')); os.chdir(root)
from offline_offline.iql import Policy, ValueFunc
from envs.mujoco.call_mujoco_env import call_mujoco_env

# positive-control check (fourth note): can the oracle v_t's own actor (target-only iql on all 402k paired transitions) hop in the target,
# and is v_t calibrated against the discounted return its own policy realizes?

hid = yaml.safe_load(open('config/mujoco/score_iql/hopper.yaml'))['hidden_sizes']
prefix = 'logs/gate_o/oracle_v/score_iql/hopper-friction-srcdatatype-medium-replay-tardatatype-medium-2.0/r0/models/model'
actor = Policy(11, 3, 1., hidden_size = hid); actor.load_state_dict(torch.load(prefix + '_actor', map_location = 'cpu')); actor.eval()
v = ValueFunc(11, 3); v.load_state_dict(torch.load(prefix + '_value', map_location = 'cpu')); v.eval()
gamma = 0.99

def episode(env):
    s, done, obs, rews = env.reset(), False, [], []
    while not done:
        with torch.no_grad():
            _, _, mean = actor(torch.tensor(s, dtype = torch.float32).view(1, -1))
        obs.append(s); s, rew, done, _ = env.step(mean.squeeze(0).numpy()); rews.append(rew)
    obs, rews = np.array(obs, dtype = np.float32), np.array(rews)
    g = np.zeros_like(rews)
    acc = 0.
    for i in range(len(rews) - 1, -1, -1):
        acc = rews[i] + gamma * acc; g[i] = acc
    with torch.no_grad():
        vals = v(torch.tensor(obs)).squeeze(-1).numpy()
    return rews.sum(), len(rews), obs[:, 5].mean(), obs[:, 0].min(), g[0], vals[0], np.abs(vals - g).mean(), np.corrcoef(vals, g)[0, 1] if len(g) > 2 else np.nan

for name, env in (('target friction 2.0', call_mujoco_env({'env_name': 'hopper-friction', 'shift_level': 2.0})), ('source Hopper-v3', gym.make('Hopper-v3'))):
    env.seed(0)
    rows = [episode(env) for _ in range(10)]
    print(f'--- v_t actor in {name}')
    print('return   len   mean vx   min z   G0 (realized)   V(s0)   mean|V - G|   corr(V, G)')
    for r in rows:
        print(f'{r[0]:7.1f}  {r[1]:4d}   {r[2]:6.2f}   {r[3]:5.2f}   {r[4]:8.1f}   {r[5]:8.1f}   {r[6]:8.1f}   {r[7]:6.2f}')
    print(f'mean: return {np.mean([r[0] for r in rows]):.1f}, len {np.mean([r[1] for r in rows]):.0f}, vx {np.mean([r[2] for r in rows]):.2f}')
