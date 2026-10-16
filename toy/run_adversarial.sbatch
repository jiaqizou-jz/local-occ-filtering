#!/bin/bash
#SBATCH -J toy_adv
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /home/jz520/local-occ-filtering/logs/toy_adversarial-%j.log
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
/home/jz520/envs/odrl/bin/python -u -m toy.adversarial
echo "EXIT=$?"
