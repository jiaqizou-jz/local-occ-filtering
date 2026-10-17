#!/bin/bash
#SBATCH -J gate_o_ref
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --array=0-2
#SBATCH -o /home/jz520/local-occ-filtering/logs/gate_o_ref-%A_%a.log
# reference / baseline runs on odrl's own target data (real trajectories in the target physics, call_tar_dataset), no score, no paired file
# MODE=concat        plain odrl iql, 128 source + 128 target per step
# MODE=target_only   score_iql with both halves from the real target data (256 target per step)
# ENV / SHIFT / SRCTYPE / OUT select the cell: gate o = hopper-friction 2.0 medium-replay (logs/gate_o/reference_*),
#                                              gate o2 = hopper-gravity 0.5 medium (logs/gate_o2/*)
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
MODE=${MODE:-concat}; SEED=$SLURM_ARRAY_TASK_ID
ENV=${ENV:-hopper-friction}; SHIFT=${SHIFT:-2.0}; SRCTYPE=${SRCTYPE:-medium-replay}; OUT=${OUT:-logs/gate_o/reference_$MODE}
hostname; nvidia-smi -L; echo "reference $MODE seed $SEED env $ENV shift $SHIFT srctype $SRCTYPE out $OUT"
if [ "$MODE" = "concat" ]; then
  /home/jz520/envs/odrl/bin/python -u train.py --policy iql --mode 3 --env $ENV --shift_level $SHIFT --srctype $SRCTYPE --tartype medium \
    --seed $SEED --max_step ${MAX_STEP:-500000} --save-model --dir $OUT
else
  /home/jz520/envs/odrl/bin/python -u train.py --policy score_iql --mode 3 --env $ENV --shift_level $SHIFT --srctype $SRCTYPE --tartype medium \
    --seed $SEED --max_step ${MAX_STEP:-500000} --save-model --dir $OUT --params "{\"source\": \"target_only\"}"
fi
echo "EXIT=$?"
