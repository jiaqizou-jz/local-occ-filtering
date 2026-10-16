import re, glob, sys, os
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/home/jz520/local-occ-filtering')
os.chdir('/home/jz520/local-occ-filtering')

# every eval prints source then target: take the target lines, score a run by the mean of its last 5 target evals
runs = defaultdict(dict)
for f in sorted(glob.glob('logs/gate_o-*_*.log')):
    txt = open(f).read()
    m = re.search(r'arm (\S+) seed (\d+) keep (\S+)', txt)
    if not m: continue
    arm, seed, keep = m.group(1), int(m.group(2)), m.group(3)
    evals = [float(x) for x in re.findall(r'Evaluation over \d+ episodes: (-?\d+\.\d+)', txt)]
    tar = evals[1::2]
    done = 'EXIT=0' in txt
    runs[(keep, arm)][seed] = dict(last5 = np.mean(tar[-5:]) if len(tar) >= 5 else np.nan, best = max(tar) if tar else np.nan, n = len(tar), done = done, curve = tar)

want_resid = '--resid' in sys.argv
if want_resid:
    from odrl.oracle import bellman_residual
print('keep  arm            seeds  evals   last5 per seed                 mean±sd     best mean   ' + ('resid' if want_resid else ''))
for (keep, arm), d in sorted(runs.items()):
    seeds = sorted(d)
    l5 = np.array([d[s]['last5'] for s in seeds]); best = np.array([d[s]['best'] for s in seeds])
    resid = ''
    if want_resid:
        rs = []
        for s in seeds:
            p = f'logs/gate_o/keep{keep}/{arm}/score_iql/hopper-friction-srcdatatype-medium-replay-tardatatype-medium-2.0/r{s}/models/model'
            try: rs.append(bellman_residual(p, device = 'cpu'))
            except Exception as e: rs.append(np.nan)
        resid = ' '.join(f'{r:.2f}' for r in rs)
    print(f'{keep:<5} {arm:<14} {len(seeds):<6} {[d[s]["n"] for s in seeds]}  {" ".join(f"{x:7.1f}" for x in l5):<30} {np.nanmean(l5):7.1f}±{np.nanstd(l5):5.1f}   {np.nanmean(best):7.1f}   {resid}')
