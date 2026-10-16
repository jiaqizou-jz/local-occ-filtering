#!/bin/bash
#SBATCH -J odrl_paired_steps
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 10:00:00
#SBATCH --array=0-2
#SBATCH -o /home/jz520/local-occ-filtering/logs/odrl_paired_steps-%A_%a.log
# power ablation on the perfect-support targets: normalized-model gain column, then classifier steps 3000 -> 20000
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
/home/jz520/envs/odrl/bin/python -u -c "
from odrl.detectability import run
for steps in (3000, 20000):
    print('=== steps', steps, flush = True)
    run(cases = ('null', 'paired_0.5', 'paired_2.0'), seeds = ($SLURM_ARRAY_TASK_ID,), prefix = 'halfcheetah', steps = steps)
"
echo "EXIT=$?"
