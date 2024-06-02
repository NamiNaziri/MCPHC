import os
import signal
import debugpy
import time
import subprocess
#debugpy.listen(5678)
#print("Waiting for debugger attach")
#debugpy.wait_for_client()
import re



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


def sort_files(directory):
    # Get all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if 'HumanoidAeMcpPnn7PPO.pth' in files:
        files.remove('HumanoidAeMcpPnn7PPO.pth')
    # Sort files by name
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    return files

directory = '/home/naminaziri/Desktop/Final_Git/MCPHC/runs/HumanoidAeMcpPnn7PPO/nn'
checkpoint_path = 'checkpoint=/home/naminaziri/Desktop/Final_Git/MCPHC/runs/HumanoidAeMcpPnn7PPO/nn/' #TODO: change this each time we want to continue the training to the last epoch
best_checkpoint = 'HumanoidAeMcpPnn7PPO.pth'

train_command = ['python', 'src/phc/train.py', 'task=HumanoidAeMcpPnn7', 'task.env.numEnvs=2', 'headless=True', 'test=False', 'train.params.config.minibatch_size=2', '+debug1=FALSE']
test_command_base = ['python', 'src/phc/train.py', 'task=HumanoidAeMcpPnn7', 'task.env.numEnvs=2', 'headless=True', 'test=True', 'train.params.config.minibatch_size=2','train.params.config.player.games_num=14', '+debug1=FALSE']

pattern = r'(\d+)'
prev_last_epoch = 0
epoch_interval = 10

train_sub = subprocess.Popen(train_command)

time_train_check_interval = 20

last_epoch = -1

while(True): 
    
    
    sorted_files = sort_files(directory)
    if(len(sorted_files) > 2):
        last_checkpoint = sorted_files[-1]
        print(last_checkpoint)
        match = re.search(pattern, last_checkpoint)
        last_epoch =  int(match.group(1)) #int(sorted_files[-1][:-4])
        
    print('-------------------------------------------------')
    print(prev_last_epoch)
    print(last_epoch)
    
    if (last_epoch - prev_last_epoch) >= epoch_interval:

        prev_last_epoch = last_epoch
        
        # Pause Training
        os.kill(train_sub.pid, signal.SIGSTOP)
        

        # Test the best checkpoint 
        test_command = test_command_base.copy()
        test_command.append(checkpoint_path + best_checkpoint)
        test_sub = subprocess.Popen(test_command)
        test_sub.wait()

        
        
        # Test the last checkpoint
        test_command = test_command_base.copy()
        test_command.append(checkpoint_path + last_checkpoint)
        test_sub = subprocess.Popen(test_command)
        test_sub.wait()

        test_sub = None

        # Resume training
        os.kill(train_sub.pid, signal.SIGCONT)
        
    print('-------------------------------------------------')
    


    

    time.sleep(time_train_check_interval)

# Print sorted files
for file in sorted_files:
    print(file)