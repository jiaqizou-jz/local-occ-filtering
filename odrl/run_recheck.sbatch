#!/bin/bash
#SBATCH -J gate_o_recheck
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 6:00:00
#SBATCH -o /home/jz520/local-occ-filtering/logs/recheck-%j.log
# fifth note: clean-subset recheck of two results on the same held-out rows (no clipped velocities, no terminal rows), training unchanged
#   behavior   same-physics false positive (abs / cabs / rep), seeds 0-2, replicate
#   paired_2.0 friction perfect-support, delta clustering K = 10: rho_dara vs rho_norm / rho_local, seeds 0-2
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
/home/jz520/envs/odrl/bin/python -u - <<'PY'
from odrl.detectability import run
run(('behavior',), seeds = (0, 1, 2), gate_p = False, replicate = True, clean = True)
run(('paired_2.0',), seeds = (0, 1, 2), axis = 'friction', gate_p = False, cluster_on = 'delta', num_clusters = 10, clean = True)
PY
echo "EXIT=$?"
