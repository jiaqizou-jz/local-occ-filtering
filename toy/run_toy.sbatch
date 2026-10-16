#!/bin/bash
#SBATCH -J toy_ratio
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /home/jz520/local-occ-filtering/logs/toy_ratio-%j.log
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/jz520/envs/dukesim/bin/python
hostname; $PY -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads())"
echo "=== experiment 0: identity test ==="
$PY -c "from toy.test_ratio_identity import test_joint_minus_occupancy_is_conditional as t; t(); print('experiment 0 pass')"
echo "=== experiment 1: four-cell matrix, 5 seeds ==="
$PY -c "from toy.evaluate_ratio import run; run(seeds = 5)"
echo "EXIT=$?"
