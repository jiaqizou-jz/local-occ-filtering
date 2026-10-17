from __future__ import annotations

import sys, os, glob
from pathlib import Path
import numpy as np, torch, yaml, h5py

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'algo')); os.chdir(root)
from offline_offline.iql import Policy
from envs.mujoco.call_mujoco_env import call_mujoco_env

# why does target-only learning give ~0 return on halfcheetah gravity 0.5 while the data policy earns 4.3 / step?
# per arm (seed 0): action mse of the saved actor on the target data (fit), and in rollout: return, pitch (obs[1]) excursions, forward speed

hid = yaml.safe_load(open('config/mujoco/score_iql/hopper.yaml'))['hidden_sizes']
for cell in ('halfcheetah-gravity', 'halfcheetah-friction'):
    env = call_mujoco_env({'env_name': cell, 'shift_level': 0.5}); env.seed(0)
    with h5py.File(f'dataset/mujoco/{cell.replace("-", "_")}_0.5_medium.hdf5') as h:
        s, a = torch.tensor(h['observations'][:]), torch.tensor(h['actions'][:])
        print(f'{cell} data: pitch |obs[1]| mean {h["observations"][:, 1].__abs__().mean():.2f} max {np.abs(h["observations"][:, 1]).max():.2f}, vx mean {h["observations"][:, 8].mean():.2f}, |a| mean {np.abs(h["actions"][:]).mean():.2f}, frac |a| > 0.99 {(np.abs(h["actions"][:]) > 0.99).mean():.2f}')
    for arm in ('target_only', 'bc', 'concat'):
        p = glob.glob(f'logs/panel/{cell.replace("-", "_")}_0.5/{arm}/*/*/r0/models/model_actor')[0]
        actor = Policy(17, 6, 1., hidden_size = hid); actor.load_state_dict(torch.load(p, map_location = 'cpu')); actor.eval()
        with torch.no_grad():
            _, _, m = actor(s); mse = ((m - a) ** 2).mean().item(); amag = m.abs().mean().item()
            R, flips, first, vx = [], [], [], []
            for _ in range(5):
                o, done, ret, t, f0, pitch, v = env.reset(), False, 0., 0, None, [], []
                while not done:
                    _, _, act = actor(torch.tensor(o, dtype = torch.float32).view(1, -1))
                    o, r, done, _ = env.step(act.squeeze(0).numpy()); ret += r; t += 1; pitch.append(o[1]); v.append(o[8])
                    if f0 is None and abs(o[1]) > 1.5: f0 = t
                R.append(ret); flips.append(np.mean(np.abs(pitch) > 1.5)); first.append(f0); vx.append(np.mean(v))
        print(f'  {arm:<12} fit mse {mse:.4f} |a| {amag:.2f} | rollout return {np.mean(R):7.1f}  frac steps |pitch| > 1.5: {np.mean(flips):.2f}  first flip step {first}  vx {np.mean(vx):.2f}')
