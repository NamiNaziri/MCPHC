#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --output=log.out

echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."

# For the next line to work, you need to be in the
# hpc-examples directory.
srun singularity run --nv --bind /scratch conda.sif python run_logging_daemon.py 