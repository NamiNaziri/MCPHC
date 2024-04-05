from argparse import ArgumentParser
import os
import signal
import time
import subprocess
#debugpy.listen(5678)
#print("Waiting for debugger attach")
#debugpy.wait_for_client()
import re

parser = ArgumentParser()
parser.add_argument("--experiment", type=str, default="")
args = parser.parse_args()

train_sub = None
test_sub = None


import atexit

def quit_gracefully():
    print('All subprocesses has been killed.')
    if  train_sub is not None:
        train_sub.kill()

    if test_sub is not None:
        test_sub.kill()


atexit.register(quit_gracefully)
pattern = r'_ep_(\d+)_rew'

def get_value(file_name):
    match = re.search(pattern, file_name)
    if match:
        return int(match.group(1))
    else:
        return -1  # If pattern not found, consider it as lowest priority


def sort_files(directory):
    # Get all files in the directory
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except FileNotFoundError:
        print("Directory does not exist.")
        return []
    
    if 'HumanoidAeMcpPnn6PPO.pth' in files:
        files.remove('HumanoidAeMcpPnn6PPO.pth')
    # Sort files by name
    #files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    files = sorted(files, key=get_value)
    return files
user = "nazirin1"

directory = f'/home/{user}/scratch/Final_Git/Final_scratch/MCPHC/runs/{args.experiment}/nn'
print(directory)
checkpoint_path = f'checkpoint=/home/{user}/scratch/Final_Git/Final_scratch/MCPHC/runs/{args.experiment}/nn/' #TODO: change this each time we want to continue the training to the last epoch
best_checkpoint = 'HumanoidAeMcpPnn6PPO.pth'

train_command = [ 'singularity', 'run', '--nv', '--bind', '/scratch', 'conda.sif','python', 'src/phc/train.py', 'task=HumanoidAeMcpPnn6', 'task.env.numEnvs=2', 'headless=True', 'test=False', 'train.params.config.minibatch_size=2', '+debug1=FALSE']
#test_command_base = ['python', 'src/phc/train.py', 'task=HumanoidAeMcpPnn6', 'task.env.numEnvs=1', 'headless=True', 'test=True', 'train.params.config.minibatch_size=1','train.params.config.player.games_num=1', '+debug1=FALSE']
test_command_base = ['/opt/conda/envs/mcphc/bin/python', 'src/phc/train.py', 'task=HumanoidAeMcpPnn6', 'task.env.numEnvs=1', 'headless=True', 'test=True', 'train.params.config.minibatch_size=1','train.params.config.player.games_num=1', '+debug1=FALSE']


prev_last_epoch = -1
epoch_interval = 75


time_train_check_interval = 20
command = 'singularity run --nv --bind /scratch conda.sif python src/phc/train.py task=HumanoidAeMcpPnn6 task.env.numEnvs=1 headless=True test=True train.params.config.minibatch_size=1 train.params.config.player.games_num=1'

last_epoch = -1

while(True): 
    
    
    sorted_files = sort_files(directory)
    print(sorted_files)
    if(len(sorted_files) > 2):
        last_checkpoint = sorted_files[-1]
        match = re.search(pattern, last_checkpoint)
        lastest_epoch =  int(match.group(1)) // epoch_interval
        print(lastest_epoch)
        if(lastest_epoch - last_epoch > 1):
            last_epoch = last_epoch + 1

            last_checkpoint = sorted_files[last_epoch] # always one epoch behind
            print(last_checkpoint)
            #match = re.search(pattern, last_checkpoint)
            #last_epoch =  int(match.group(1)) #int(sorted_files[-1][:-4])
    print('-------------------------------------------------')
    print(prev_last_epoch)
    print(last_epoch)
    
    if (last_epoch - prev_last_epoch) > 0:

        prev_last_epoch = last_epoch

        # Test the best checkpoint 
        test_command = test_command_base.copy()
        test_command.append(checkpoint_path + last_checkpoint)
        #os.system(command + f" checkpoint={last_checkpoint}" + ' +debug1=False')
        test_sub = subprocess.Popen(  test_command)
        test_sub.wait()
        #time.sleep(time_test)

        #test_sub.kill()
        
    print('-------------------------------------------------')
    


    

    time.sleep(time_train_check_interval)

# Print sorted files
for file in sorted_files:
    print(file)