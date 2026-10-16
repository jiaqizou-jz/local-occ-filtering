import sys, os, h5py, numpy as np, torch
sys.path.insert(0, '/home/jz520/local-occ-filtering'); sys.path.insert(0, '/home/jz520/local-occ-filtering/algo'); os.chdir('/home/jz520/local-occ-filtering')
from offline_offline.iql import DoubleQFunc, ValueFunc
with h5py.File('dataset/mujoco/hopper_friction_2.0_pairedheld.hdf5', 'r') as h:
    s, a = torch.tensor(h['observations'][:]), torch.tensor(h['actions'][:])
run = 'logs/gate_o/keep0.5/{arm}/score_iql/hopper-friction-srcdatatype-medium-replay-tardatatype-medium-2.0/r{seed}/models/model'
print('arm/seed           Q min/mean/max              V min/mean/max')
for arm in ('concat', 'random', 'oracle_bellman', 'oracle_state', 'dara', 'target_only'):
    for seed in (0, 1, 2):
        q = DoubleQFunc(11, 3); q.load_state_dict(torch.load(run.format(arm = arm, seed = seed) + '_critic', map_location = 'cpu')); q.eval()
        v = ValueFunc(11, 3); v.load_state_dict(torch.load(run.format(arm = arm, seed = seed) + '_value', map_location = 'cpu')); v.eval()
        with torch.no_grad():
            qm = torch.min(*q(s, a)).squeeze(-1); vv = v(s).squeeze(-1)
        print(f'{arm + "/" + str(seed):<18} {qm.min():9.1f} {qm.mean():9.1f} {qm.max():9.1f}      {vv.min():9.1f} {vv.mean():9.1f} {vv.max():9.1f}')
