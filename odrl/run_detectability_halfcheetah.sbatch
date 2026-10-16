#!/bin/bash
#SBATCH -J hc_detect
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH --array=0-2
#SBATCH -o /home/jz520/local-occ-filtering/logs/hc_detectability-%A_%a.log
# halfcheetah gate s + p at h = 1 (gate r passed for g2.0 at h = 1, k = 10)
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
/home/jz520/envs/odrl/bin/python -u -c "
from odrl.detectability import run
run(cases = ('null', 'behavior', 'behavior_m', 'g0.5_medium', 'g0.5_random', 'g2.0_medium', 'g2.0_random'), seeds = ($SLURM_ARRAY_TASK_ID,), prefix = 'halfcheetah')
"
echo "EXIT=$?"
