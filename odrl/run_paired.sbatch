#!/bin/bash
#SBATCH -J odrl_paired
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH --array=0-2
#SBATCH -o /home/jz520/local-occ-filtering/logs/odrl_paired-%A_%a.log
# perfect-support targets: source (s, a) stepped in the target physics, hopper then halfcheetah, gate p + gate c
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
/home/jz520/envs/odrl/bin/python -u -c "
from odrl.detectability import run
for prefix in ('hopper', 'halfcheetah'):
    print('=== prefix', prefix, flush = True)
    run(cases = ('null', 'paired_0.5', 'paired_2.0'), seeds = ($SLURM_ARRAY_TASK_ID,), prefix = prefix)
"
echo "EXIT=$?"
