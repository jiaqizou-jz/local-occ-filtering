from __future__ import annotations

import sys, os, re, glob
from pathlib import Path
import numpy as np, torch, yaml, h5py

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'algo')); os.chdir(root)
from offline_offline.iql import Policy, DoubleQFunc
from envs.mujoco.call_mujoco_env import call_mujoco_env
from envs.infos import REF_MIN_SCORE, REF_MAX_SCORE

# second batch of the sixth note: (1) bc along n_t, (2) lambda sweep at n_t 5000 / 1000, (3) six-cell panel
# job ids from logs/gate_o_jobs.txt; primary = mean of the last 5 target evals, secondary = target episode length / forward speed, q mean on target data

jobs = dict(re.findall(r'(bc_nt|lambda|panel) (\d+)', [l for l in open('logs/gate_o_jobs.txt') if l.startswith('batch2 bc_nt')][-1]))
hid = yaml.safe_load(open('config/mujoco/score_iql/hopper.yaml'))['hidden_sizes']
DIMS = {'hopper': (11, 3, 5), 'halfcheetah': (17, 6, 8), 'walker2d': (17, 6, 8)}   # obs, act, index of forward velocity in obs
envs, data = {}, {}

def cell(name, shift = 0.5):
    if name not in envs:
        e = call_mujoco_env({'env_name': name, 'shift_level': shift}); e.seed(0); envs[name] = e
        with h5py.File(f'dataset/mujoco/{name.replace("-", "_")}_{shift}_medium.hdf5', 'r') as h:
            data[name] = torch.tensor(h['observations'][:]), torch.tensor(h['actions'][:])
    return envs[name], data[name]

def norm(x, name, shift = 0.5):
    k = f'{name}-{shift}'
    return (x - REF_MIN_SCORE[k]) / (REF_MAX_SCORE[k] - REF_MIN_SCORE[k]) * 100

def evals(f):
    e = [float(x) for x in re.findall(r'Evaluation over \d+ episodes: (-?\d+\.\d+)', open(f).read())]
    return np.array(e[0::2]), np.array(e[1::2])

@torch.no_grad()
def rollout(prefix, name, n = 5):
    env, _ = cell(name); so, ao, vi = DIMS[name.split('-')[0]]
    actor = Policy(so, ao, 1., hidden_size = hid); actor.load_state_dict(torch.load(prefix + '_actor', map_location = 'cpu')); actor.eval()
    L, V = [], []
    for _ in range(n):
        s, done, t, vx = env.reset(), False, 0, []
        while not done:
            _, _, m = actor(torch.tensor(s, dtype = torch.float32).view(1, -1))
            s, r, done, _ = env.step(m.squeeze(0).numpy()); t += 1; vx.append(s[vi])
        L.append(t); V.append(np.mean(vx))
    return np.mean(L), np.mean(V)

@torch.no_grad()
def qmag(prefix, name):
    _, (ts, ta) = cell(name); so, ao, _ = DIMS[name.split('-')[0]]
    q = DoubleQFunc(so, ao); q.load_state_dict(torch.load(prefix + '_critic', map_location = 'cpu')); q.eval()
    return torch.min(*q(ts, ta)).squeeze(-1).numpy().mean()

def run(log, pattern, name):
    src, tar = evals(log); p = glob.glob(pattern)[0][:-len('_critic')]
    L, vx = rollout(p, name)
    return dict(tar = tar[-5:].mean(), src = src[-5:].mean(), L = L, vx = vx, q = qmag(p, name), n = len(tar))

def row(label, r, name):
    return f'{label:<34} {r["tar"]:8.1f}  {norm(r["tar"], name):5.1f}   {r["src"]:8.1f}   {r["L"]:5.0f}  {r["vx"]:5.2f}  {r["q"]:11.1f}  (n eval {r["n"]})'

def stats(xs):
    xs = np.array(xs); return f'{xs.mean():7.1f} ± {xs.std():5.1f}'

def wins(a, b, c):
    # arm a 'wins' against arms b and c: mean above both by more than the larger sd of b / c, and 3/3 seeds above c
    a, b, c = map(np.array, (a, b, c))
    return a.mean() > max(b.mean(), c.mean()) + max(b.std(), c.std()) and (a > c).all()

HOP = 'hopper-gravity'
first = {}   # first-batch reference arms (iql target_only / concat along n_t) by (nt, arm, seed) -> target last-5
for i in range(18):
    nt, arm, s = (500, 1000, 2000)[i // 6], ('target_only', 'concat')[(i // 3) % 2], i % 3
    first[(nt, arm, s)] = evals(f'logs/nt_sweep-202868_{i}.log')[1][-5:].mean()
for s in range(3):
    first[(5000, 'target_only', s)] = evals(f'logs/gate_o_ref-202845_{s}.log')[1][-5:].mean()
    first[(5000, 'concat', s)] = evals(f'logs/gate_o_ref-202844_{s}.log')[1][-5:].mean()
    first[(5000, 'bc', s)] = evals(f'logs/nt_sweep-202869_{s}.log')[1][-5:].mean()

print('=== (1) bc along n_t (hopper gravity 0.5)')
print(f'{"run":<34} {"target":>8}  {"norm":>5}   {"source":>8}   {"eplen":>5}  {"vx":>5}  {"q mean":>11}')
bc = {}
for i in range(9):
    nt, s = (500, 1000, 2000)[i // 3], i % 3
    r = run(f'logs/nt_sweep-{jobs["bc_nt"]}_{i}.log', f'logs/gate_o2/nt{nt}/bc/*/*/r{s}/models/model_critic', HOP); bc[(nt, s)] = r['tar']
    print(row(f'bc nt{nt} seed {s}', r, HOP))
for s in range(3): bc[(5000, s)] = first[(5000, 'bc', s)]
print()
print('n_t     bc mean±sd        iql target_only    concat            bc - target_only per seed      bc - concat per seed')
for nt in (500, 1000, 2000, 5000):
    b = [bc[(nt, s)] for s in range(3)]; t = [first[(nt, 'target_only', s)] for s in range(3)]; c = [first[(nt, 'concat', s)] for s in range(3)]
    print(f'{nt:<7} {stats(b)}   {stats(t)}   {stats(c)}   ' + ' / '.join(f'{x - y:+7.0f}' for x, y in zip(b, t)) + '      ' + ' / '.join(f'{x - y:+7.0f}' for x, y in zip(b, c)))

print()
print('=== (2) lambda sweep (hopper gravity 0.5), L = (1 - lambda) mean_target + lambda mean_source')
lam = {}
for i in range(24):
    l, nt, s = (0, 0.125, 0.25, 0.5)[i // 6], (5000, 1000)[(i // 3) % 2], i % 3
    r = run(f'logs/lam_sweep-{jobs["lambda"]}_{i}.log', f'logs/lambda/nt{nt}/lam{l}/*/*/r{s}/models/model_critic', HOP); lam[(l, nt, s)] = r['tar']
    print(row(f'lambda {l} nt{nt} seed {s}', r, HOP))
print()
print('n_t    lambda   mean±sd          per seed                    wins vs lambda 0 and 1/2?')
for nt in (5000, 1000):
    for l in (0, 0.125, 0.25, 0.5):
        xs = [lam[(l, nt, s)] for s in range(3)]
        w = wins(xs, [lam[(0, nt, s)] for s in range(3)], [lam[(0.5, nt, s)] for s in range(3)]) if l not in (0, 0.5) else '-'
        print(f'{nt:<6} {l:<8} {stats(xs)}   ' + ' / '.join(f'{x:7.1f}' for x in xs) + f'        {w}')
    print(f'{nt:<6} reference: target_only {stats([first[(nt, "target_only", s)] for s in range(3)])}  concat {stats([first[(nt, "concat", s)] for s in range(3)])}')

print()
print('=== (3) six-cell panel, 5000 target transitions, medium source')
CELLS = ('hopper-friction', 'halfcheetah-gravity', 'halfcheetah-friction', 'walker2d-gravity', 'walker2d-friction'); ARMS = ('target_only', 'concat', 'bc')
panel = {(HOP, arm, s): first[(5000, arm, s)] for arm in ARMS for s in range(3)}
for i in range(45):
    c, arm, s = CELLS[i // 9], ARMS[(i // 3) % 3], i % 3
    r = run(f'logs/panel-{jobs["panel"]}_{i}.log', f'logs/panel/{c.replace("-", "_")}_0.5/{arm}/*/*/r{s}/models/model_critic', c); panel[(c, arm, s)] = r['tar']
    print(row(f'{c} {arm} seed {s}', r, c))
print()
print('cell                   target_only          concat               bc                   delta t-c per seed             delta bc-t per seed            verdict')
for c in (HOP,) + CELLS:
    t, co, b = ([panel[(c, a, s)] for s in range(3)] for a in ARMS)
    d = np.array(t) - np.array(co)
    verdict = 'target_only' if (d > 0).all() and abs(d.mean()) > max(np.std(t), np.std(co)) else 'concat' if (d < 0).all() and abs(d.mean()) > max(np.std(t), np.std(co)) else 'tie'
    print(f'{c:<22} {stats(t)} ({norm(np.mean(t), c):5.1f})  {stats(co)} ({norm(np.mean(co), c):5.1f})  {stats(b)} ({norm(np.mean(b), c):5.1f})  ' + ' / '.join(f'{x:+7.0f}' for x in d) + '      ' + ' / '.join(f'{x - y:+7.0f}' for x, y in zip(b, t)) + f'      {verdict}')
