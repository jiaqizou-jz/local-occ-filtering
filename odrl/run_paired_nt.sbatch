#!/bin/bash
#SBATCH -J odrl_paired_nt
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 10:00:00
#SBATCH --array=0-2
#SBATCH -o /home/jz520/local-occ-filtering/logs/odrl_paired_nt-%A_%a.log
# n_t sweep on the one configuration where ranking is possible: hopper friction 2.0 perfect support, delta clusters k = 10
# question: is there an n_t where the cluster estimators beat the global continuous one (dara)?
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
/home/jz520/envs/odrl/bin/python -u -c "
from odrl.detectability import run
for nt in (500, 1000, 2500, 5000, 10000, 20000):
    print('=== num_target', nt, flush = True)
    run(cases = ('null', 'paired_2.0'), seeds = ($SLURM_ARRAY_TASK_ID,), prefix = 'hopper', axis = 'friction', cluster_on = 'delta', num_clusters = 10, num_target = nt)
"
echo "EXIT=$?"
