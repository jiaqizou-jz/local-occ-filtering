from __future__ import annotations

import sys, os, re, glob
from pathlib import Path
import numpy as np, torch, gym, yaml, h5py

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'algo')); os.chdir(root)
from offline_offline.iql import Policy, DoubleQFunc, ValueFunc
from envs.mujoco.call_mujoco_env import call_mujoco_env
from envs.infos import get_normalized_score

# gate o2 (fifth note): hopper gravity 0.5 medium, target_only vs concat, 3 seeds. primary = mean of the last 5 target evals (fixed before launch);
# secondary = target episode length, forward speed (saved actor, deterministic mean action), q / v magnitude on the real target data

hid = yaml.safe_load(open('config/mujoco/score_iql/hopper.yaml'))['hidden_sizes']
env = call_mujoco_env({'env_name': 'hopper-gravity', 'shift_level': 0.5}); env.seed(0)
with h5py.File('dataset/mujoco/hopper_gravity_0.5_medium.hdf5', 'r') as h:
    ts, ta = torch.tensor(h['observations'][:]), torch.tensor(h['actions'][:])

def evals(f):
    e = [float(x) for x in re.findall(r'Evaluation over \d+ episodes: (-?\d+\.\d+)', open(f).read())]
    return np.array(e[0::2]), np.array(e[1::2])

def rollout(actor, n = 5):
    out = []
    for _ in range(n):
        s, done, t, ret, vx = env.reset(), False, 0, 0., []
        while not done:
            with torch.no_grad(): _, _, m = actor(torch.tensor(s, dtype = torch.float32).view(1, -1))
            s, r, done, _ = env.step(m.squeeze(0).numpy()); t += 1; ret += r; vx.append(s[5])
        out.append((t, ret, np.mean(vx)))
    return np.mean([o[0] for o in out]), np.mean([o[1] for o in out]), np.mean([o[2] for o in out])

print('arm          seed  target last5  norm   source last5   ep len   rollout ret   vx    Q mean/max (target data)   V mean')
rows = {}
for arm, jobs in (('target_only', '202845'), ('concat', '202844')):
    for s in range(3):
        src, tar = evals(f'logs/gate_o_ref-{jobs}_{s}.log')
        p = glob.glob(f'logs/gate_o2/{arm}/*/*/r{s}/models/model_critic')[0][:-len('_critic')]
        actor = Policy(11, 3, 1., hidden_size = hid); actor.load_state_dict(torch.load(p + '_actor', map_location = 'cpu')); actor.eval()
        q = DoubleQFunc(11, 3); q.load_state_dict(torch.load(p + '_critic', map_location = 'cpu')); q.eval()
        v = ValueFunc(11, 3); v.load_state_dict(torch.load(p + '_value', map_location = 'cpu')); v.eval()
        with torch.no_grad(): qm, vm = torch.min(*q(ts, ta)).squeeze(-1).numpy(), v(ts).squeeze(-1).numpy()
        L, R, vx = rollout(actor)
        t5 = tar[-5:].mean(); rows[(arm, s)] = t5
        print(f'{arm:<12} {s}     {t5:7.1f}    {get_normalized_score(t5, "hopper-gravity-0.5"):5.1f}   {src[-5:].mean():7.1f}       {L:5.0f}    {R:7.1f}     {vx:5.2f}   {qm.mean():8.1f} {qm.max():8.1f}        {vm.mean():8.1f}   (n eval {len(tar)})')
for arm in ('target_only', 'concat'):
    x = np.array([rows[(arm, s)] for s in range(3)]); print(f'{arm:<12} mean {x.mean():7.1f} ± {x.std():5.1f}')
print('per-seed difference target_only - concat: ' + ' / '.join(f'{rows[("target_only", s)] - rows[("concat", s)]:+.0f}' for s in range(3)))
