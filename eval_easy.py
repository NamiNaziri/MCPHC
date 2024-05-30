import shutil
import os
import argparse
import subprocess
import debugpy

if __name__ == "__main__":
    # Specify the paths of the source and destination directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="")
    parser.add_argument("--debug1", type=bool, default=False)
    args = parser.parse_args()

    # train_command = ['singularity', 'run', '--nv', '--bind', '/scratch', 'conda.sif','python', f'triton/{args.experiment}/phc/train.py', f'train.params.config.full_experiment_name="{args.experiment}', 'task=HumanoidAeMcpPnn6', 'task.env.numEnvs=512', 'headless=True', 'test=False', 'train.params.config.minibatch_size=8192', '+debug1=FALSE']
    # test_command =  ['singularity', 'run', '--nv', '--bind', '/scratch', 'opengl.sif','python', 'new_run_copy.py ', '--experiment', f'{args.experiment}', f'train.params.config.full_experiment_name="{args.experiment}']

    test_command = f"python triton/{args.experiment}/phc/train.py task=HumanoidAeMcpPnn6 task.env.numEnvs=2 headless=False test=True train.params.config.minibatch_size=2 checkpoint=/home/naminaziri/scratch2/AGit/MCPHC/runs/{args.experiment}/nn/HumanoidAeMcpPnn6PPO.pth +debug1={args.debug1} +pythonpath={args.experiment}"

    # thread.start_new_thread(os.system, train_command)
    # thread.start_new_thread(os.system, test_command)
    os.system(test_command)

    # train_sub = subprocess.Popen(  train_command)
