#!/bin/bash
#SBATCH -J b2_summary
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o /home/jz520/local-occ-filtering/logs/b2_summary-%j.log
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=8 D4RL_SUPPRESS_IMPORT_ERROR=1
/home/jz520/envs/odrl/bin/python -u odrl/batch2_summary.py 2>&1 | grep -v -E "Warn|warn|load_state"
echo "EXIT=$?"
