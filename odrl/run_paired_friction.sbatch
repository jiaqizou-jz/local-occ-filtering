#!/bin/bash
#SBATCH -J odrl_paired_fric
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH --array=0-2
#SBATCH -o /home/jz520/local-occ-filtering/logs/odrl_paired_friction-%A_%a.log
# gain-type axis: perfect-support friction targets, hopper then halfcheetah, gate p + gate c
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
/home/jz520/envs/odrl/bin/python -u -c "
from odrl.detectability import run
for prefix in ('hopper', 'halfcheetah'):
    print('=== prefix', prefix, flush = True)
    run(cases = ('null', 'paired_0.5', 'paired_2.0'), seeds = ($SLURM_ARRAY_TASK_ID,), prefix = prefix, axis = 'friction')
"
echo "EXIT=$?"
