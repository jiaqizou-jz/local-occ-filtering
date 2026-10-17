#!/bin/bash
#SBATCH -J panel
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --array=0-44
#SBATCH -o /home/jz520/local-occ-filtering/logs/panel-%A_%a.log
# sixth note, second batch: six-cell panel {hopper, halfcheetah, walker2d} x {gravity 0.5, friction 0.5}, medium source (d4rl *-medium-v2),
#   odrl medium target (5000 transitions); arms target_only / concat / bc, 3 seeds. hopper gravity 0.5 is the existing gate o2 + bc runs,
#   so array 0-44 covers the other five cells: cell = id / 9, arm = (id / 3) % 3, seed = id % 3
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
CELLS=(hopper-friction halfcheetah-gravity halfcheetah-friction walker2d-gravity walker2d-friction); ARMS=(target_only concat bc)
CELL=${CELLS[$((SLURM_ARRAY_TASK_ID / 9))]}; ARM=${ARMS[$(((SLURM_ARRAY_TASK_ID / 3) % 3))]}; SEED=$((SLURM_ARRAY_TASK_ID % 3))
if [ "$ARM" = "bc" ]; then PARAMS='{"source": "target_only", "bc": true}'; else PARAMS="{\"source\": \"$ARM\"}"; fi
OUT=${OUT_PREFIX:-logs/panel}/${CELL/-/_}_0.5/$ARM
hostname; nvidia-smi -L; echo "panel cell $CELL arm $ARM seed $SEED out $OUT"
/home/jz520/envs/odrl/bin/python -u train.py --policy score_iql --mode 3 --env $CELL --shift_level 0.5 --srctype medium --tartype medium \
  --seed $SEED --max_step ${MAX_STEP:-500000} --save-model --dir $OUT --params "$PARAMS"
echo "EXIT=$?"
