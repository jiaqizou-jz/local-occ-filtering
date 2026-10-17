#!/bin/bash
#SBATCH -J nt_sweep
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --array=0-17
#SBATCH -o /home/jz520/local-occ-filtering/logs/nt_sweep-%A_%a.log
# sixth note, hopper gravity 0.5 medium (source d4rl hopper-medium-v2, target odrl 5000 real transitions)
#   first batch, array 0-17: n_t in {500, 1000, 2000} x arm in {target_only, concat} x seed in {0, 1, 2}; nested random subsets,
#               subset seed = training seed, both arms share the subset; the n_t = 5000 point is the existing gate o2 runs (same protocol)
#   BC=1: target-only behaviour cloning (same actor, action mse, advantage weight 1); BC_NT=5000 array 0-2 (first batch, all 5000),
#               default array 0-8: n_t in {500, 1000, 2000} x seed (second batch, same nested subsets as the iql arms)
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
SEED=$((SLURM_ARRAY_TASK_ID % 3)); PREFIX=${OUT_PREFIX:-logs/gate_o2}; NTS=(500 1000 2000)
if [ "${BC:-0}" = "1" ]; then
  ARM=bc
  if [ "${BC_NT:-sweep}" = "5000" ]; then
    NT=5000; PARAMS="{\"source\": \"target_only\", \"bc\": true}"; OUT=$PREFIX/bc
  else
    NT=${NTS[$((SLURM_ARRAY_TASK_ID / 3))]}
    PARAMS="{\"source\": \"target_only\", \"bc\": true, \"target_subset\": $NT, \"subset_seed\": $SEED}"; OUT=$PREFIX/nt$NT/bc
  fi
else
  ARMS=(target_only concat)
  NT=${NTS[$((SLURM_ARRAY_TASK_ID / 6))]}; ARM=${ARMS[$(((SLURM_ARRAY_TASK_ID / 3) % 2))]}
  PARAMS="{\"source\": \"$ARM\", \"target_subset\": $NT, \"subset_seed\": $SEED}"; OUT=$PREFIX/nt$NT/$ARM
fi
hostname; nvidia-smi -L; echo "nt_sweep arm $ARM nt $NT seed $SEED out $OUT"
/home/jz520/envs/odrl/bin/python -u train.py --policy score_iql --mode 3 --env hopper-gravity --shift_level 0.5 --srctype medium --tartype medium \
  --seed $SEED --max_step ${MAX_STEP:-500000} --save-model --dir $OUT --params "$PARAMS"
echo "EXIT=$?"
