#!/bin/bash
#SBATCH -J lam_sweep
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --array=0-23
#SBATCH -o /home/jz520/local-occ-filtering/logs/lam_sweep-%A_%a.log
# sixth note, second batch: relative loss weight L = (1 - lambda) mean_target + lambda mean_source (v / q / actor share lambda)
#   hopper gravity 0.5 medium; array 0-23: lambda in {0, 1/8, 1/4, 1/2} x n_t in {5000, 1000} x seed in {0, 1, 2}
#   n_t = 5000 is the full odrl target set (gate o2 protocol); n_t = 1000 is the nested subset of the first batch (subset seed = training seed)
#   lambda = 1/2 is the plain concat mean; lambda = 0 keeps only the 128 target samples of each 128 + 128 batch
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
SEED=$((SLURM_ARRAY_TASK_ID % 3)); LAMS=(0 0.125 0.25 0.5); NTS=(5000 1000); PREFIX=${OUT_PREFIX:-logs/lambda}
LAM=${LAMS[$((SLURM_ARRAY_TASK_ID / 6))]}; NT=${NTS[$(((SLURM_ARRAY_TASK_ID / 3) % 2))]}
if [ "$NT" = "5000" ]; then SUB=""; else SUB=", \"target_subset\": $NT, \"subset_seed\": $SEED"; fi
PARAMS="{\"source\": \"concat\", \"source_weight\": $LAM$SUB}"; OUT=$PREFIX/nt$NT/lam$LAM
hostname; nvidia-smi -L; echo "lam_sweep lambda $LAM nt $NT seed $SEED out $OUT"
/home/jz520/envs/odrl/bin/python -u train.py --policy score_iql --mode 3 --env hopper-gravity --shift_level 0.5 --srctype medium --tartype medium \
  --seed $SEED --max_step ${MAX_STEP:-500000} --save-model --dir $OUT --params "$PARAMS"
echo "EXIT=$?"
