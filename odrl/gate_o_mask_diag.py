import sys, os, h5py, numpy as np
sys.path.insert(0, '/home/jz520/local-occ-filtering'); os.chdir('/home/jz520/local-occ-filtering')
from odrl.oracle import source_dataset

d = source_dataset('hopper')
z = np.load('logs/oracle/hopper_friction_2.0_scores.npz')
s, ns, r, term, a = d['observations'], d['next_observations'], d['rewards'], d['terminals'], d['actions']
scale = s.std(0) + 1e-6
S, NS = (s / scale).astype(np.float32), (ns / scale).astype(np.float32)
rng = np.random.default_rng(0)

# closure: how far is a kept s' from the nearest kept s (standardized coords)
def nn_dist(q, ref, chunk = 128):
    ref_sq = (ref ** 2).sum(-1)
    out = []
    for i in range(0, len(q), chunk):
        x = q[i:i + chunk]
        d2 = (x ** 2).sum(-1)[:, None] + ref_sq[None] - 2. * x @ ref.T
        out.append(np.sqrt(np.maximum(d2.min(-1), 0.)))
    return np.concatenate(out)

print(f'source n {len(s)}, terminal frac {term.mean():.4f}, reward mean {r.mean():.3f}, z mean {s[:, 0].mean():.3f}, vx mean {s[:, 5].mean():.3f}')
print('arm            keep_n   term_frac  rew   z      vx     |a|    gap_mean  nn_med  nn_p90  nn_p99  frac>1')
ref_full = rng.choice(len(s), 3000, replace = False)
dist_full = nn_dist(NS[ref_full], S)
print(f'{"concat":<14} {len(s):>7}  {term.mean():.4f}   {r.mean():.2f}  {s[:,0].mean():.3f}  {s[:,5].mean():.3f}  {np.abs(a).mean():.3f}  {z["gap"].mean():.3f}     {np.median(dist_full):.3f}   {np.percentile(dist_full, 90):.3f}   {np.percentile(dist_full, 99):.3f}   {(dist_full > 1).mean():.3f}')
for key in ('random', 'oracle_state', 'dara', 'oracle_bellman'):
    sc = z[key]; keep = sc >= np.quantile(sc, 0.5); idx = np.flatnonzero(keep)
    q = rng.choice(idx, 3000, replace = False)
    dist = nn_dist(NS[q], S[idx])
    print(f'{key:<14} {len(idx):>7}  {term[keep].mean():.4f}   {r[keep].mean():.2f}  {s[keep,0].mean():.3f}  {s[keep,5].mean():.3f}  {np.abs(a[keep]).mean():.3f}  {z["gap"][keep].mean():.3f}     {np.median(dist):.3f}   {np.percentile(dist, 90):.3f}   {np.percentile(dist, 99):.3f}   {(dist > 1).mean():.3f}')

# the paired target set used by target_only: transition level, no trajectories
with h5py.File('dataset/mujoco/hopper_friction_2.0_paired.hdf5', 'r') as h:
    ts, tns, tterm, tr = h['observations'][:], h['next_observations'][:], h['terminals'][:], h['rewards'][:]
TS, TNS = (ts / scale).astype(np.float32), (tns / scale).astype(np.float32)
dist = nn_dist(TNS[rng.choice(len(ts), 3000, replace = False)], TS)
print(f'{"target_only":<14} {len(ts):>7}  {tterm.mean():.4f}   {tr.mean():.2f}  {ts[:,0].mean():.3f}  {ts[:,5].mean():.3f}  {"-":>5}  {"-":>5}     {np.median(dist):.3f}   {np.percentile(dist, 90):.3f}   {np.percentile(dist, 99):.3f}   {(dist > 1).mean():.3f}')

# overlap between masks
masks = {k: z[k] >= np.quantile(z[k], 0.5) for k in ('random', 'oracle_state', 'dara', 'oracle_bellman')}
print('mask overlap (jaccard): ' + ', '.join(f'{a}/{b} {(masks[a] & masks[b]).sum() / (masks[a] | masks[b]).sum():.2f}' for a, b in (('oracle_state', 'dara'), ('oracle_state', 'oracle_bellman'), ('dara', 'oracle_bellman'), ('random', 'oracle_state'))))
