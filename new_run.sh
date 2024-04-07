#!/bin/bash


#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"

echo "$1"
experiment_name=$1
echo ""$experiment_name".pi"

echo "Hello $USER! You are on node $HOSTNAME.  The time is $(date)."

# For the next line to work, you need to be in the
# hpc-examples directory.
#srun python3 train_eval_combo.py --experiment="$experiment_name"

srun singularity run --nv --bind /scratch opengl.sif python triton/$experiment_name/phc/train.py train.params.config.full_experiment_name="$experiment_name"  task=HumanoidAeMcpPnn6 task.env.numEnvs=512 headless=True test=False train.params.config.minibatch_size=8192 +debug1=False  +pythonpath=$experiment_name
#srun singularity run --nv --bind /scratch opengl.sif python new_run_copy.py --experiment="$experiment_name"

