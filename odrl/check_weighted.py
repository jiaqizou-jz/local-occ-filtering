from __future__ import annotations

import sys, os, copy
from pathlib import Path
import numpy as np, torch, yaml

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'algo')); os.chdir(root)
from utils import ReplayBuffer
from offline_offline.score_iql import ScoreIQL

# lambda = 1/2 of the weighted path must reproduce the parent's plain concat update on the same batch, for every network;
# lambda = 0 must equal the parent's update on a batch made of the target half only. parameter differences after one adam step

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = yaml.safe_load(open('config/mujoco/score_iql/hopper.yaml'))
config.update(state_dim = 11, action_dim = 3, max_action = 1., device = device)

def buffer(n, seed):
    rng = np.random.default_rng(seed); b = ReplayBuffer(11, 3, device)
    b.state, b.action, b.next_state = (rng.standard_normal((n, d)).astype(np.float32) for d in (11, 3, 11))
    b.reward, b.not_done = rng.standard_normal((n, 1)).astype(np.float32), (rng.random((n, 1)) > 0.1).astype(np.float32)
    b.size = n
    return b

def agent(**over):
    torch.manual_seed(0); a = ScoreIQL({**config, **over}, device); a.prepared = True
    return a

def params(a):
    return torch.cat([p.detach().flatten() for net in (a.v_func, a.q_funcs, a.policy) for p in net.parameters()])

src, tar = buffer(1000, 1), buffer(1000, 2)
plain, half = agent(), agent(source_weight = 0.5)
assert torch.equal(params(plain), params(half))
np.random.seed(0); plain.train(src, tar, 128)
np.random.seed(0); half.train(src, tar, 128)
d_half = (params(plain) - params(half)).abs().max().item()
step = (params(plain) - params(agent())).abs().max().item()
print(f'lambda 1/2 vs plain concat: max |param diff| {d_half:.2e} (step size {step:.2e})')

# lambda = 0 vs parent trained on the target half only (both halves of the batch drawn from the same 128 target rows)
zero = agent(source_weight = 0.)
np.random.seed(0); zero.train(src, tar, 128)
np.random.seed(0); _ = src.sample(128); ind = np.random.randint(0, tar.size, size = 128)   # replay the rng to recover the target rows
sub = buffer(128, 3); sub.state, sub.action, sub.next_state, sub.reward, sub.not_done = (getattr(tar, k)[ind] for k in ('state', 'action', 'next_state', 'reward', 'not_done'))
ref = agent()
class Fixed:                                   # returns the whole 128-row target subset every call
    size = 128
    def sample(self, n): return tuple(torch.FloatTensor(getattr(sub, k)).to(device) for k in ('state', 'action', 'next_state', 'reward', 'not_done'))
ref.train(Fixed(), Fixed(), 128)
d_zero = (params(zero) - params(ref)).abs().max().item()
print(f'lambda 0 vs plain path on the target half only: max |param diff| {d_zero:.2e}')
assert d_half < 1e-2 * step and d_zero < 1e-2 * step, (d_half, d_zero, step)   # fp32 rounding of the zero-weighted rows shows up in adam at ~1e-6
print('check ok')
