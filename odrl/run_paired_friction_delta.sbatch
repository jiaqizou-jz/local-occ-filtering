#!/bin/bash
#SBATCH -J odrl_fric_delta
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH --array=0-2
#SBATCH -o /home/jz520/local-occ-filtering/logs/odrl_fric_delta-%A_%a.log
# the one axis where ranking is possible (hopper friction 2.0, rho_dara 0.68): give the cluster channel a representation with resolution
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
/home/jz520/envs/odrl/bin/python -u -c "
from odrl.detectability import run
for cluster_on, k in (('delta', 10), ('delta', 30), ('next', 30)):
    print('=== cluster_on', cluster_on, 'k', k, flush = True)
    run(cases = ('null', 'paired_2.0'), seeds = ($SLURM_ARRAY_TASK_ID,), prefix = 'hopper', axis = 'friction', cluster_on = cluster_on, num_clusters = k)
"
echo "EXIT=$?"
