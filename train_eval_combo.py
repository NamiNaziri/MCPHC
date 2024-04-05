import shutil
import os
import argparse
import subprocess

if __name__ == "__main__":
    # Specify the paths of the source and destination directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="")
    args = parser.parse_args()
    #train_command = ['singularity', 'run', '--nv', '--bind', '/scratch', 'conda.sif','python', f'triton/{args.experiment}/phc/train.py', f'train.params.config.full_experiment_name="{args.experiment}', 'task=HumanoidAeMcpPnn6', 'task.env.numEnvs=512', 'headless=True', 'test=False', 'train.params.config.minibatch_size=8192', '+debug1=FALSE']
    #test_command =  ['singularity', 'run', '--nv', '--bind', '/scratch', 'opengl.sif','python', 'new_run_copy.py ', '--experiment', f'{args.experiment}', f'train.params.config.full_experiment_name="{args.experiment}']

    train_command = f'singularity run --nv --bind /scratch conda.sif python triton/{args.experiment}/phc/train.py train.params.config.full_experiment_name="$experiment_name"  task=HumanoidAeMcpPnn6 task.env.numEnvs=512 headless=True test=False train.params.config.minibatch_size=8192 +debug1=False'
    test_command = f'singularity run --nv --bind /scratch opengl.sif python new_run_copy.py --experiment {args.experiment}'
    train_command += " &"
    test_command += " &"

    #thread.start_new_thread(os.system, train_command)
    #thread.start_new_thread(os.system, test_command)
    os.system(train_command)
    os.system(test_command)

    #train_sub = subprocess.Popen(  train_command)
