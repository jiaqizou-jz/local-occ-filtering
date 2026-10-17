#!/bin/bash
#SBATCH -J gate_o_ref
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --array=0-2
#SBATCH -o /home/jz520/local-occ-filtering/logs/gate_o_ref-%A_%a.log
# gate o positive control (fourth note): odrl's own target data (real trajectories in the target physics, call_tar_dataset) instead of the paired set
# MODE=concat   plain odrl iql, half source / half target batches
# MODE=target_only   score_iql with both halves from the real target data, no score, no paired file
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
MODE=${MODE:-concat}; SEED=$SLURM_ARRAY_TASK_ID
hostname; nvidia-smi -L; echo "reference $MODE seed $SEED"
if [ "$MODE" = "concat" ]; then
  /home/jz520/envs/odrl/bin/python -u train.py --policy iql --mode 3 --env hopper-friction --shift_level 2.0 --srctype medium-replay --tartype medium \
    --seed $SEED --max_step ${MAX_STEP:-500000} --save-model --dir logs/gate_o/reference_concat
else
  /home/jz520/envs/odrl/bin/python -u train.py --policy score_iql --mode 3 --env hopper-friction --shift_level 2.0 --srctype medium-replay --tartype medium \
    --seed $SEED --max_step ${MAX_STEP:-500000} --save-model --dir logs/gate_o/reference_target_only --params "{\"source\": \"target_only\"}"
fi
echo "EXIT=$?"
