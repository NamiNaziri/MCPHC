from dataclasses import dataclass
from typing import List

import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym import gymutil
from omegaconf import OmegaConf
from rl_games.common import vecenv, env_configurations
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner
from torch import nn
from torch.distributions import Normal
from tqdm import tqdm

from phc.env.tasks import humanoid_amp_task
from phc.env.tasks.vec_task_wrappers import VecTaskPythonWrapper
from phc.learning import amp_players, amp_agent, amp_models, amp_network_builder, amp_network_mcp_builder, amp_network_pnn_builder, im_amp, im_amp_players
from phc.learning.amp_models import ModelAMPContinuous
from phc.learning.amp_network_mcp_builder import AMPMCPBuilder
from phc.learning.amp_players import rescale_actions
from phc.learning.pnn import PNN
from phc.utils.flags import flags

import torch
import yaml
from scipy.spatial.transform import Rotation as sRot

from phc.env.tasks.humanoid_im import HumanoidIm, compute_imitation_observations_v7
from phc.learning.amp_network_builder import AMPBuilder
from phc.utils.config import load_cfg, parse_sim_params, get_args
from phc.utils.motion_lib_smpl import MotionLibSMPL
from phc.utils.parse_task import parse_task
from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree

args = None
cfg = None
cfg_train = None


def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)

        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)

    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print(env.num_envs)
    print(env.num_actions)
    print(env.num_obs)
    print(env.num_states)

    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUAlgoObserver(AlgoObserver):

    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return


class RLGPUEnv(vecenv.IVecEnv):

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space

        info['enc_amp_observation_space'] = self.env.enc_amp_observation_space

        if isinstance(self.env.task, humanoid_amp_task.HumanoidAMPTask):
            info['task_obs_size'] = self.env.task.get_task_obs_size()
        else:
            info['task_obs_size'] = 0

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info


def load_pnn(mlp_args, state_dict_load, num_prim, has_lateral, activation="relu", device="cpu"):
    net_key_name = "actors.0"
    loading_keys = [k for k in state_dict_load.keys() if k.startswith(net_key_name) and k.endswith('bias')]
    layer_size = []
    for idx, key in enumerate(loading_keys):
        layer_size.append(state_dict_load[key].shape[::-1][0])

    pnn = PNN(mlp_args, output_size=layer_size[-1], numCols=num_prim, has_lateral=has_lateral)
    pnn.load_state_dict(state_dict_load)

    pnn.freeze_pnn(num_prim)
    pnn.to(device)
    return pnn


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs), 'vecenv_type': 'RLGPU'})


def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)
    return runner


def main():
    global args
    global cfg
    global cfg_train

    args = get_args()
    args.cfg_env = "src/phc/data/cfg/phc_kp_mcp_iccv.yaml"
    args.cfg_train = "src/phc/data/cfg/train/rlg/HumanoidAeMcpPnnPPO.yaml"
    args.headless = False
    args.test = True
    args.task = "HumanoidAeMcpPnn"
    args.num_envs = 1

    cfg = OmegaConf.load(args.cfg_env)
    cfg_train = OmegaConf.load(args.cfg_train)

    motion_lib_args = torch.load("data/theirs/motion_lib_args.pkl")
    motion_lib: MotionLibSMPL = torch.load("data/theirs/their_motion_lib.pkl")
    motion_lib_kwargs = motion_lib.kwargs
    motion_lib_kwargs["motion_file"] = "/home/nhgk/scratch/workspace/PerpetualHumanoidControl/data/amass/pkls/amass_copycat_take5_train.pkl"
    motion_lib: MotionLibSMPL = motion_lib.__class__(**motion_lib_kwargs)

    flags.debug, flags.follow, flags.fixed, flags.divide_group, flags.no_collision_check, flags.fixed_path, flags.real_path, flags.small_terrain, flags.show_traj, flags.server_mode, flags.slow, flags.real_traj, flags.im_eval, flags.no_virtual_display, flags.render_o3d = \
        args.debug, args.follow, False, False, False, False, False, args.small_terrain, True, args.server_mode, False, False, args.im_eval, args.no_virtual_display, args.render_o3d

    flags.add_proj = args.add_proj
    flags.has_eval = args.has_eval
    flags.trigger_input = False
    flags.demo = args.demo

    algo_observer = RLGPUAlgoObserver()

    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()

    vargs = vars(args)
    runner.run(vargs)


if __name__ == "__main__":
    import pydevd_pycharm

    pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)
    main()
