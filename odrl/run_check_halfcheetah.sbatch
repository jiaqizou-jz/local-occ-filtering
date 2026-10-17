#!/bin/bash
#SBATCH -J hc_check
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 0:30:00
#SBATCH -o /home/jz520/local-occ-filtering/logs/hc_check-%j.log
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
/home/jz520/envs/odrl/bin/python -u odrl/check_halfcheetah.py 2>&1 | grep -v -E "Warn|warn"
echo "EXIT=$?"
