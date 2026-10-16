#!/bin/bash
#SBATCH -J odrl_gate_s_rep
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --array=0-2
#SBATCH -o /home/jz520/local-occ-filtering/logs/odrl_gate_s_rep-%A_%a.log
# leak vs noise diagnostic for the on-support residual: signed mean + spearman between two independent fits on the same held-out points
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
/home/jz520/envs/odrl/bin/python -u -c "
from odrl.detectability import run
run(cases = ('null', 'behavior'), seeds = ($SLURM_ARRAY_TASK_ID,), gate_p = False, replicate = True)
"
echo "EXIT=$?"
