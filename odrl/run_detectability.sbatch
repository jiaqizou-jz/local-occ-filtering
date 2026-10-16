#!/bin/bash
#SBATCH -J odrl_detect
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH --array=0-2
#SBATCH -o /home/jz520/local-occ-filtering/logs/odrl_detectability-%A_%a.log
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
/home/jz520/envs/odrl/bin/python -u -c "
from odrl.detectability import run
run(seeds = ($SLURM_ARRAY_TASK_ID,))
"
echo "EXIT=$?"
