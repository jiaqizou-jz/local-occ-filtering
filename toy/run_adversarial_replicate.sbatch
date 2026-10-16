#!/bin/bash
#SBATCH -J toy_adv_rep
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /home/jz520/local-occ-filtering/logs/toy_adversarial_replicate-%j.log
# confirmatory replication of the post hoc experiment 1b criterion on fresh seeds (preregistered in docs/PLAN.md before launch)
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
/home/jz520/envs/odrl/bin/python -u -c "
from toy.adversarial import run
run(seeds = (3, 4, 5))
"
echo "EXIT=$?"
