#!/bin/bash
#SBATCH -J b2_smoke
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 1:00:00
#SBATCH --array=0-4
#SBATCH -o /home/jz520/local-occ-filtering/logs/b2_smoke-%A_%a.log
# second batch smoke: 0 = weighted-path equivalence check; 1 = bc nt500; 2 = lambda 1/8 nt1000; 3 = walker2d friction target_only; 4 = halfcheetah gravity bc
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
case $SLURM_ARRAY_TASK_ID in
  0) /home/jz520/envs/odrl/bin/python -u odrl/check_weighted.py ;;
  1) BC=1 OUT_PREFIX=logs/b2_smoke MAX_STEP=3000 SLURM_ARRAY_TASK_ID=0 bash odrl/run_nt_sweep.sbatch ;;
  2) OUT_PREFIX=logs/b2_smoke MAX_STEP=3000 SLURM_ARRAY_TASK_ID=9 bash odrl/run_lambda_sweep.sbatch ;;
  3) OUT_PREFIX=logs/b2_smoke MAX_STEP=3000 SLURM_ARRAY_TASK_ID=36 bash odrl/run_panel.sbatch ;;
  4) OUT_PREFIX=logs/b2_smoke MAX_STEP=3000 SLURM_ARRAY_TASK_ID=15 bash odrl/run_panel.sbatch ;;
esac
echo "SMOKE_EXIT=$?"
