#!/bin/bash
#SBATCH -J oracle_v
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH -o /home/jz520/local-occ-filtering/logs/oracle_v-%j.log
# evaluation oracle v_t: target-only iql on every source (s, a) stepped in the target physics (pairedbig), used only to score source transitions
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
/home/jz520/envs/odrl/bin/python -u train.py --policy score_iql --mode 3 --env hopper-friction --shift_level 2.0 \
  --srctype medium-replay --tartype medium --seed 0 --max_step ${MAX_STEP:-500000} --save-model \
  --dir logs/gate_o/oracle_v \
  --params '{"source": "target_only", "target_file": "dataset/mujoco/hopper_friction_2.0_pairedbig.hdf5"}'
echo "EXIT=$?"
