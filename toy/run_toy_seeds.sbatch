#!/bin/bash
#SBATCH -J toy_seed
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH --array=0-2
#SBATCH -o /home/jz520/local-occ-filtering/logs/toy_seed-%A_%a.log
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
/home/jz520/envs/odrl/bin/python -u -c "
import time
from toy.evaluate_ratio import evaluate_case
seed = $SLURM_ARRAY_TASK_ID
keys = None
for name in 'ABCD':
    t0 = time.time()
    r = evaluate_case(name, seed = seed)
    if keys is None:
        keys = [k for k in r if k != 'case']
        print('seed  case  ' + ''.join(f'{k:>10}' for k in keys), flush = True)
    print(f'{seed:<6}{r[\"case\"]:<6}' + ''.join(f'{r[k]:>10.3f}' for k in keys) + f'   ({time.time() - t0:.0f}s)', flush = True)
"
echo "EXIT=$?"
