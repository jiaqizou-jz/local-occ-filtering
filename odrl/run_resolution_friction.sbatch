#!/bin/bash
#SBATCH -J fric_resolution
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o /home/jz520/local-occ-filtering/logs/fric_resolution-%j.log
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname
/home/jz520/envs/odrl/bin/python -u -c "
from odrl.resolution import run
for prefix in ('hopper', 'halfcheetah'):
    print('=== prefix', prefix, flush = True)
    run(source = prefix + '-medium-replay-v2', env = prefix + '-friction', horizons = (1, 5, 20), num_clusters = (10, 30, 100, 300, 1000))
"
echo "EXIT=$?"
