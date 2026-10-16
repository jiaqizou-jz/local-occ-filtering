#!/bin/bash
#SBATCH -J gate_o_smoke
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o /home/jz520/local-occ-filtering/logs/gate_o_smoke-%j.log
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
for ARM in oracle_state target_only; do
  echo "== arm $ARM"; start=$(date +%s)
  /home/jz520/envs/odrl/bin/python -u train.py --policy score_iql --mode 3 --env hopper-friction --shift_level 2.0 \
    --srctype medium-replay --tartype medium --seed 0 --max_step 20000 --save-model --dir logs/smoke/gate_o/$ARM \
    --params "{\"source\": \"$ARM\", \"keep_fraction\": 0.5, \"score_file\": \"logs/oracle/hopper_friction_2.0_scores.npz\", \"target_file\": \"dataset/mujoco/hopper_friction_2.0_paired.hdf5\"}" 2>&1 | grep -v "load datafile\|Warning\|warnings.warn"
  echo "arm $ARM: $(( $(date +%s) - start )) s for 20000 steps"
done
/home/jz520/envs/odrl/bin/python -u -c "
from odrl.oracle import bellman_residual
print('bellman residual (oracle_state smoke model):', bellman_residual('logs/smoke/gate_o/oracle_state/score_iql/hopper-friction-srcdatatype-medium-replay-tardatatype-medium-2.0/r0/models/model'))
" 2>&1 | grep -v Warning
echo "EXIT=$?"
