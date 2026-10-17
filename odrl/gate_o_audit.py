from __future__ import annotations

import sys, os
from pathlib import Path
import numpy as np
import gym, d4rl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); os.chdir(Path(__file__).resolve().parent.parent)
from odrl.oracle import source_dataset, paired, ORACLE_DIR

# gate o audit (fourth note): the state gap was ||s'_target,sim - s'_source,DATA||; d4rl.qlearning_dataset builds s'_source,DATA = observations[i + 1],
# which at a terminal row is the next episode's reset state. re-simulate every source transition in the source physics and compare per category.

d = source_dataset('hopper')
s, a, ns, r, term = (d[k] for k in ('observations', 'actions', 'next_observations', 'rewards', 'terminals'))
raw = gym.make('hopper-medium-replay-v2').get_dataset()
N = len(raw['rewards'])
keep = np.array([i for i in range(N - 1) if not raw['timeouts'][i]])
assert len(keep) == len(s), (len(keep), len(s))
assert np.allclose(raw['observations'][keep], s, atol = 1e-5) and np.allclose(raw['actions'][keep], a, atol = 1e-5)
raw_next = raw['next_observations'][keep].astype(np.float32)   # the raw key that qlearning_dataset ignores
print(f'raw rows {N}, timeouts {int(raw["timeouts"].sum())}, terminals {int(raw["terminals"].sum())}; qlearning rows {len(s)}, terminals {int(term.sum())}; observations[i+1] == raw next_observations on {np.isclose(ns, raw_next, atol = 1e-5).all(-1).mean():.4f} of rows', flush = True)

ns_sim, r_sim, d_sim = paired(gym.make('Hopper-v3'), s, a)
scale = ns.std(0) + 1e-6
z = dict(np.load(ORACLE_DIR / 'hopper_friction_2.0_scores.npz'))
ns_t = z['target_next_state']
gap_data, gap_sim = z['gap'], np.linalg.norm((ns_t - ns_sim) / scale, axis = -1)
err_data = np.linalg.norm((ns - ns_sim) / scale, axis = -1)
err_raw = np.linalg.norm((raw_next - ns_sim) / scale, axis = -1)
clipped = (np.abs(s[:, 5:]) >= 10 - 1e-6).any(-1) | (np.abs(ns[:, 5:]) >= 10 - 1e-6).any(-1)
t = term.astype(bool)
keep_data, keep_sim = -gap_data >= np.quantile(-gap_data, 0.5), -gap_sim >= np.quantile(-gap_sim, 0.5)
rank = lambda x: np.argsort(np.argsort(x))

q = lambda x: f'{np.median(x):7.3f} {np.percentile(x, 90):7.3f} {x.max():9.3f}'
print('category               n       |s_data - s_sim|  med/p90/max   |raw_next - s_sim| med/p90/max   |r - r_sim| med/max   d != d_sim   gap_data med   gap_sim med   kept@50% data   kept@50% sim')
for name, m in (('nonterminal', ~t & ~clipped), ('nonterminal clipped', ~t & clipped), ('terminal', t & ~clipped), ('terminal clipped', t & clipped), ('all', np.ones_like(t))):
    if m.sum() == 0:
        continue
    print(f'{name:<20} {m.sum():>8}   {q(err_data[m])}   {q(err_raw[m])}   {np.median(np.abs(r - r_sim)[m]):.4f} {np.abs(r - r_sim)[m].max():.3f}   {(d_sim[m] != term[m]).mean():.4f}   {np.median(gap_data[m]):.3f}   {np.median(gap_sim[m]):.3f}   {keep_data[m].mean():.4f}   {keep_sim[m].mean():.4f}')
print(f'spearman(gap_data, gap_sim) all {np.corrcoef(rank(gap_data), rank(gap_sim))[0, 1]:.4f}, nonterminal {np.corrcoef(rank(gap_data[~t]), rank(gap_sim[~t]))[0, 1]:.4f}; terminals kept at 50%: data {int((keep_data & t).sum())}, sim {int((keep_sim & t).sum())}; gap_sim mean {gap_sim.mean():.3f} cv {gap_sim.std() / gap_sim.mean():.2f}', flush = True)

z.update(oracle_state_sim = (-gap_sim).astype(np.float32), gap_sim = gap_sim.astype(np.float32), source_sim_next_state = ns_sim, source_sim_reward = r_sim, source_sim_done = d_sim)
np.savez(ORACLE_DIR / 'hopper_friction_2.0_scores.npz', **z)
print('scores written: oracle_state_sim', flush = True)
