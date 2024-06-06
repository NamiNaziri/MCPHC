import time

import os
import time
import subprocess

test_command_base = ['python', 'eval_easy.py', '--experiment', ]

def refresh_sshfs_cache(directory):
    """Refresh the sshfs cache by listing the directory contents.
    
    This function forces sshfs to update its cache by recursively listing
    all files and directories within the specified directory. This helps
    ensure that any changes made on the remote filesystem are reflected
    locally, which is important for detecting new files and directories.
    """
    try:
        # List the contents of the directory recursively to refresh the cache.
        # The output is suppressed to avoid cluttering the console.
        subprocess.run(['ls', '-lR', directory], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error refreshing sshfs cache: {e}")

def get_directories_and_files(parent_directory):
    """Returns a dictionary with directories as keys and their files as values."""
    directory_contents = {}
    for root, dirs, files in os.walk(parent_directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            file_list = os.listdir(dir_path)
            file_list.sort(key=lambda x: os.path.getmtime(os.path.join(dir_path, x)))  # Sort files by modification time
            directory_contents[dir_path] = file_list
            #directory_contents[dir_path] = set(os.listdir(dir_path))
    return directory_contents

def monitor_directory(parent_directory, polling_interval=10):
    """Monitors the directory for new folders and files."""
    previous_state = get_directories_and_files(parent_directory)
    
    print(f"Monitoring directory: {parent_directory}")
    
    while True:
        time.sleep(polling_interval)
        
        # Refresh the sshfs cache before checking
        refresh_sshfs_cache(parent_directory)
        
        current_state = get_directories_and_files(parent_directory)

        # Check for new directories
        new_directories = set(current_state.keys()) - set(previous_state.keys())
        for new_dir in new_directories:
            print(f"New directory created: {new_dir}")
        
        # Check for new files in existing directories
        for directory in current_state:
            if directory in previous_state:
                new_files = current_state[directory] - previous_state[directory]
                print(new_files)
                for new_file in new_files:
                    if ".pth" in new_file:
                        parts = directory.split('/')
                        print(f"New file created in {directory}:*Start* {os.path.join(directory, new_file)}")
                        experiment = parts[1]
                        try:
                            subprocess.run(['python', 'eval_easy_logger.py', '--experiment', experiment, '--checkpoint', new_file], check=True, shell=False,)
                            
                            # p = subprocess.Popen(['python', 'eval_easy_logger.py', '--experiment', experiment, '--checkpoint', new_file], stdout=subprocess.PIPE, stderr=subprocess. STDOUT, shell=True)
                            # p.wait()
                        except subprocess.CalledProcessError as e:
                            print(f"Error executing eval_easy.py: {e}")

                    print(f"New file created in {directory}:*Completed* {os.path.join(directory, new_file)}")

        previous_state = current_state

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python monitor_directory.py <path_to_parent_directory>")
        sys.exit(1)
    
    parent_directory = sys.argv[1]
    
    if not os.path.isdir(parent_directory):
        print(f"The path {parent_directory} is not a valid directory.")
        sys.exit(1)
    
    monitor_directory(parent_directory)