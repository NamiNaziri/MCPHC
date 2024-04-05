from argparse import ArgumentParser
import os
from collections import deque

from inotify_simple import INotify, flags
import debugpy


command = 'singularity run --nv --bind /scratch conda.sif python src/phc/train.py task=HumanoidAeMcpPnn6 task.env.numEnvs=1 headless=True test=True train.params.config.minibatch_size=1 train.params.config.player.games_num=1'
# checkpoint=/home/naminaziri/Desktop/Final_Git/MCPHC/runs/HumanoidAeMcpPnn6PPO/nn/72.pth
#train.params.config.full_experiment_name="test"


def main():
    if args.debug1:
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    inotify = INotify()
    watch_flags = flags.CLOSE_WRITE
    root_dir = 'runs'

    # recurse into all subdirectories, adding each into add_watch
    dir_queue = deque()
    dir_queue.append(root_dir)
    wd_path_dict = {}
    while dir_queue:
        current_dir = dir_queue.popleft()
        wd = inotify.add_watch(current_dir, watch_flags)
        wd_path_dict[wd] = current_dir
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                dir_queue.append(item_path)

    print(wd_path_dict)

    while True:
        # And see the corresponding events:
        for event in inotify.read():
            print(event)
            for flag in flags.from_mask(event.mask):
                print('    ' + str(flag))
            if os.path.splitext(event.name)[1] == '.pth':
                in_posrot_path = os.path.join(wd_path_dict[event.wd], event.name)
                print(in_posrot_path)
                print(os.system("pwd"))
                os.system(command + f" checkpoint={in_posrot_path}" + ' +debug1=False')


if __name__ == '__main__':
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace(
    #     "localhost", port=12346, stdoutToServer=True, stderrToServer=True, suspend=False
    # )
    parser = ArgumentParser()
    parser.add_argument("--debug1", type=bool)
    args = parser.parse_args()

    main()
