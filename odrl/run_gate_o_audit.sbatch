#!/bin/bash
#SBATCH -J gate_o_audit
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o /home/jz520/local-occ-filtering/logs/gate_o_audit-%j.log
# re-simulate every source transition in the source physics; terminal-row gap audit; writes the corrected oracle_state_sim score
cd /home/jz520/local-occ-filtering
export OMP_NUM_THREADS=4 D4RL_SUPPRESS_IMPORT_ERROR=1
/home/jz520/envs/odrl/bin/python -u odrl/gate_o_audit.py
echo "EXIT=$?"
