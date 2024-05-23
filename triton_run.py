import shutil
import os
import argparse
import subprocess

def copy_directory(source_dir, destination_dir):
    # Create the destination directory if it doesn't exist
    current_directory = os.getcwd()
    source_dir = os.path.abspath(os.path.join(current_directory, source_dir))
    destination_dir = os.path.abspath(os.path.join(current_directory, destination_dir))
    print(source_dir)
    print(destination_dir)


    #if not os.path.exists(destination_dir):
    #    os.makedirs(destination_dir)

    # Copy the contents of the source directory to the destination directory
    try:
        shutil.copytree(source_dir, destination_dir)
        print("Directory copied successfully!")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    # Specify the paths of the source and destination directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="")
    args = parser.parse_args()
    source_directory = "./src"
    destination_directory = f'./triton/{args.experiment}'
    # Call the function to copy the directory
    copy_directory(source_directory, destination_directory)

    triton_command = ['sbatch',f'--output=out/{args.experiment}.out',f'--job-name={args.experiment}', 'new_run.sh', ]
    triton_command.append(args.experiment)
    triton_sub = subprocess.Popen(  triton_command)
    triton_sub.wait()
    