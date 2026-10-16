#!/bin/bash
#SBATCH -J odrl_gate_s
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --array=0-2
#SBATCH -o /home/jz520/local-occ-filtering/logs/odrl_gate_s-%A_%a.log
# step 1 of the 2026-10-16 plan: normalized conditional estimator vs detached residual on the same-physics rows, overlap-restricted
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
/home/jz520/envs/odrl/bin/python -u -c "
from odrl.detectability import run
run(cases = ('null', 'behavior', 'g2.0_random'), seeds = ($SLURM_ARRAY_TASK_ID,), gate_p = False)
"
echo "EXIT=$?"
