import shutil
import os
import argparse
import subprocess
import debugpy

if __name__ == "__main__":
    # Specify the paths of the source and destination directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--video", type=bool, default=False)
    parser.add_argument("--debug1", type=bool, default=False)
    args = parser.parse_args()

    # train_command = ['singularity', 'run', '--nv', '--bind', '/scratch', 'conda.sif','python', f'triton/{args.experiment}/phc/train.py', f'train.params.config.full_experiment_name="{args.experiment}', 'task=HumanoidAeMcpPnn6', 'task.env.numEnvs=512', 'headless=True', 'test=False', 'train.params.config.minibatch_size=8192', '+debug1=FALSE']
    # test_command =  ['singularity', 'run', '--nv', '--bind', '/scratch', 'opengl.sif','python', 'new_run_copy.py ', '--experiment', f'{args.experiment}', f'train.params.config.full_experiment_name="{args.experiment}']
    if(args.video):
        test_command = f"python triton/{args.experiment}/phc/train.py task=HumanoidAeMcpPnn6 task.env.numEnvs=1 train.params.config.player.games_num=1 headless=True test=True train.params.config.minibatch_size=1 checkpoint=runs/{args.experiment}/nn/{args.checkpoint}.pth +debug1={args.debug1} +pythonpath={args.experiment} task.env.episodeLength=2000"
    else:
        test_command = f"python triton/{args.experiment}/phc/train.py task=HumanoidAeMcpPnn6 task.env.numEnvs=2 headless=False test=True train.params.config.minibatch_size=2 checkpoint=runs/{args.experiment}/nn/{args.checkpoint}.pth +debug1={args.debug1} +pythonpath={args.experiment} task.env.episodeLength=3500"

    # thread.start_new_thread(os.system, train_command)
    # thread.start_new_thread(os.system, test_command)
    os.system(test_command)

    # train_sub = subprocess.Popen(  train_command)
