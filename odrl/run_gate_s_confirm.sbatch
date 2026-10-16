#!/bin/bash
#SBATCH -J odrl_gate_s_conf
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --array=3-5
#SBATCH -o /home/jz520/local-occ-filtering/logs/odrl_gate_s_conf-%A_%a.log
# confirmatory rerun of the centered in-band criterion on fresh seeds (preregistered in docs/PLAN.md)
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
/home/jz520/envs/odrl/bin/python -u -c "
from odrl.detectability import run
run(cases = ('null', 'behavior'), seeds = ($SLURM_ARRAY_TASK_ID,), gate_p = False, replicate = True)
"
echo "EXIT=$?"
