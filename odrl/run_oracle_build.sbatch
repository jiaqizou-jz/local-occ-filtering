#!/bin/bash
#SBATCH -J oracle_build
#SBATCH -p defq
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /home/jz520/local-occ-filtering/logs/oracle_build-%j.log
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
hostname; nvidia-smi -L
/home/jz520/envs/odrl/bin/python -u -c "
from odrl.oracle import build
build()
"
echo "EXIT=$?"
