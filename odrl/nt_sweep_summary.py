from __future__ import annotations

import sys, os, re, glob
from pathlib import Path
import numpy as np, torch, yaml, h5py

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'algo')); os.chdir(root)
from offline_offline.iql import Policy, DoubleQFunc
from envs.mujoco.call_mujoco_env import call_mujoco_env

# first batch of the sixth note: n_t sweep (nested target subsets) target_only vs concat, paired by seed, plus target-only bc at 5000
# primary = mean of the last 5 target evals; secondary = target episode length / forward speed (saved actor), q magnitude on the target data

hid = yaml.safe_load(open('config/mujoco/score_iql/hopper.yaml'))['hidden_sizes']
env = call_mujoco_env({'env_name': 'hopper-gravity', 'shift_level': 0.5}); env.seed(0)
norm = lambda x: (x + 26.3360015397715) / (3234.3 + 26.3360015397715) * 100
with h5py.File('dataset/mujoco/hopper_gravity_0.5_medium.hdf5', 'r') as h:
    ts, ta = torch.tensor(h['observations'][:]), torch.tensor(h['actions'][:])

def evals(f):
    e = [float(x) for x in re.findall(r'Evaluation over \d+ episodes: (-?\d+\.\d+)', open(f).read())]
    return np.array(e[0::2]), np.array(e[1::2])

def rollout(prefix, n = 5):
    actor = Policy(11, 3, 1., hidden_size = hid); actor.load_state_dict(torch.load(prefix + '_actor', map_location = 'cpu')); actor.eval()
    L, V = [], []
    for _ in range(n):
        s, done, t, vx = env.reset(), False, 0, []
        while not done:
            with torch.no_grad(): _, _, m = actor(torch.tensor(s, dtype = torch.float32).view(1, -1))
            s, r, done, _ = env.step(m.squeeze(0).numpy()); t += 1; vx.append(s[5])
        L.append(t); V.append(np.mean(vx))
    return np.mean(L), np.mean(V)

def qmag(prefix):
    q = DoubleQFunc(11, 3); q.load_state_dict(torch.load(prefix + '_critic', map_location = 'cpu')); q.eval()
    with torch.no_grad(): return torch.min(*q(ts, ta)).squeeze(-1).numpy().mean()

# (n_t, arm) -> per-seed (log, model glob)
runs = {}
for i in range(18):
    nt, arm, s = (500, 1000, 2000)[i // 6], ('target_only', 'concat')[(i // 3) % 2], i % 3
    runs[(nt, arm, s)] = (f'logs/nt_sweep-202868_{i}.log', f'logs/gate_o2/nt{nt}/{arm}/*/*/r{s}/models/model_critic')
for s in range(3):
    runs[(5000, 'target_only', s)] = (f'logs/gate_o_ref-202845_{s}.log', f'logs/gate_o2/target_only/*/*/r{s}/models/model_critic')
    runs[(5000, 'concat', s)] = (f'logs/gate_o_ref-202844_{s}.log', f'logs/gate_o2/concat/*/*/r{s}/models/model_critic')
    runs[(5000, 'bc', s)] = (f'logs/nt_sweep-202869_{s}.log', f'logs/gate_o2/bc/*/*/r{s}/models/model_critic')

res = {}
print('n_t    arm          seed  target last5   norm   source last5   ep len   vx     Q mean')
for (nt, arm, s), (log, pat) in sorted(runs.items()):
    src, tar = evals(log); p = glob.glob(pat)[0][:-len('_critic')]
    L, vx = rollout(p); res[(nt, arm, s)] = tar[-5:].mean()
    print(f'{nt:<6} {arm:<12} {s}     {tar[-5:].mean():7.1f}     {norm(tar[-5:].mean()):5.1f}    {src[-5:].mean():7.1f}      {L:5.0f}   {vx:5.2f}   {qmag(p):7.1f}   (n eval {len(tar)})')
print()
print('n_t    target_only mean±sd      concat mean±sd        paired delta per seed (target_only - concat)   norm delta')
for nt in (500, 1000, 2000, 5000):
    t = np.array([res[(nt, 'target_only', s)] for s in range(3)]); c = np.array([res[(nt, 'concat', s)] for s in range(3)])
    print(f'{nt:<6} {t.mean():7.1f} ± {t.std():5.1f}        {c.mean():7.1f} ± {c.std():5.1f}        ' + ' / '.join(f'{d:+7.0f}' for d in t - c) + f'      {np.mean([norm(x) - norm(y) for x, y in zip(t, c)]):+5.1f}')
b = np.array([res[(5000, 'bc', s)] for s in range(3)]); t = np.array([res[(5000, 'target_only', s)] for s in range(3)])
print(f'bc 5000  {b.mean():7.1f} ± {b.std():5.1f} (norm {norm(b.mean()):.1f}); bc - iql target_only per seed: ' + ' / '.join(f'{d:+.0f}' for d in b - t))
