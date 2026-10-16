#!/bin/bash
#SBATCH -J gate_o
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --array=0-14
#SBATCH -o /home/jz520/local-occ-filtering/logs/gate_o-%A_%a.log
# gate o pilot: hopper friction 2.0, perfect-support target (5000 paired transitions), iql, one variable = source selection
# arms x seeds from the array index; ARMS can be overridden from the environment (e.g. ARMS="oracle_bellman" for the late arm)
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
ARMS=(${ARMS:-target_only concat random oracle_state dara})
SEEDS=(0 1 2)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
KEEP=${KEEP:-0.5}
hostname; nvidia-smi -L; echo "arm $ARM seed $SEED keep $KEEP"
/home/jz520/envs/odrl/bin/python -u train.py --policy score_iql --mode 3 --env hopper-friction --shift_level 2.0 \
  --srctype medium-replay --tartype medium --seed $SEED --max_step ${MAX_STEP:-500000} --save-model \
  --dir logs/gate_o/keep$KEEP/$ARM \
  --params "{\"source\": \"$ARM\", \"keep_fraction\": $KEEP, \"score_file\": \"logs/oracle/hopper_friction_2.0_scores.npz\", \"target_file\": \"dataset/mujoco/hopper_friction_2.0_paired.hdf5\"}"
echo "EXIT=$?"
