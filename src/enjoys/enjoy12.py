from dataclasses import dataclass
from typing import List

import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym import gymutil
from rl_games.common import vecenv, env_configurations
from torch import nn
from torch.distributions import Normal
from tqdm import tqdm

from phc.env.tasks.vec_task_wrappers import VecTaskPythonWrapper
from phc.learning.amp_models import ModelAMPContinuous
from phc.learning.amp_network_mcp_builder import AMPMCPBuilder
from phc.learning.amp_players import rescale_actions
from phc.learning.pnn import PNN
from phc.run import RLGPUEnv, create_rlgpu_env
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
# from typing import Tuple

@dataclass
class IsaacS0:
    root_states: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor


class Rollouter:

    def __init__(
            self,
            env: VecTaskPythonWrapper,
            skeleton_trees: List[SkeletonTree],
    ):
        self.env = env
        self.skeleton_trees = skeleton_trees

    def random_action_rollout(self, s_0: IsaacS0, n_steps: int):
        if s_0 is not None:
            self.env.reset(None)
            self.env.task._humanoid_root_states[:] = s_0.root_states
            self.env.task._dof_pos[:] = s_0.dof_pos
            self.env.task._dof_vel[:] = s_0.dof_vel
            self.env.task._reset_env_tensors(None)
            self.env.task._refresh_sim_tensors()
            self.env.task.gym.simulate(self.env.task.sim)
            self.env.task._reset_env_tensors(None)
            self.env.task._refresh_sim_tensors()
        obs = self.env.task._compute_humanoid_obs(None)

        # Get some Gaussian-distributed noise
        noise = torch.tanh(np.exp(-1) * Normal(0, 1).sample((1, 10)).cuda())

        # Perform rollout
        for t in tqdm(range(n_steps)):
            self.env.step(noise)

        return IsaacS0(
            root_states=self.env.task._humanoid_root_states,
            dof_pos=self.env.task._dof_pos,
            dof_vel=self.env.task._dof_vel,
        )



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


def main():
    args = get_args()
    args.cfg_env = "phc/data/cfg/phc_kp_mcp_iccv.yaml"
    args.cfg_train = "phc/data/cfg/train/rlg/im_mcp.yaml"
    args.headless = False
    args.test = True
    args.task = "HumanoidAeMcpPnn"
    args.num_envs = 1

    cfg, cfg_train, logdir = load_cfg(args)
    motion_lib_args = torch.load("data/theirs/motion_lib_args.pkl")
    motion_lib: MotionLibSMPL = torch.load("data/theirs/their_motion_lib.pkl")
    motion_lib_kwargs = motion_lib.kwargs
    motion_lib_kwargs["motion_file"] = "/home/nhgk/scratch/workspace/PerpetualHumanoidControl/data/amass/pkls/amass_copycat_take5_train.pkl"
    motion_lib: MotionLibSMPL = motion_lib.__class__(**motion_lib_kwargs)

    # ref_d = torch.load("their_ciov7_args.pkl")
    # ref_body_pos = ref_d["ref_body_pos"]

    flags.debug, flags.follow, flags.fixed, flags.divide_group, flags.no_collision_check, flags.fixed_path, flags.real_path, flags.small_terrain, flags.show_traj, flags.server_mode, flags.slow, flags.real_traj, flags.im_eval, flags.no_virtual_display, flags.render_o3d = \
        args.debug, args.follow, False, False, False, False, False, args.small_terrain, True, args.server_mode, False, False, args.im_eval, args.no_virtual_display, args.render_o3d

    flags.add_proj = args.add_proj
    flags.has_eval = args.has_eval
    flags.trigger_input = False
    flags.demo = args.demo

    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    rollouter = Rollouter(env, motion_lib_args["skeleton_trees"])

    motion_lib.load_motions(random_sample=False, start_idx=1, max_len=-1, **motion_lib_args)
    motion_length = motion_lib.get_motion_length(0).item()
    dt = 1 / 60
    motion_res = motion_lib.get_motion_state(torch.tensor([0], device=motion_lib._device), torch.tensor([0 * dt], device=motion_lib._device), offset=None)
    s_0 = IsaacS0(
        root_states=torch.cat([motion_res["root_pos"], motion_res["root_rot"], motion_res["root_vel"], motion_res["root_ang_vel"]], dim=-1),
        dof_pos=motion_res["dof_pos"],
        dof_vel=motion_res["dof_vel"],
    )

    # motion_res = motion_lib.get_motion_state(torch.tensor([0], device=motion_lib._device), torch.tensor([100 * dt], device=motion_lib._device), offset=None)
    # g = IsaacS0(
    #     root_states=torch.cat([motion_res["root_pos"], motion_res["root_rot"], motion_res["root_vel"], motion_res["root_ang_vel"]], dim=-1),
    #     dof_pos=motion_res["dof_pos"],
    #     dof_vel=motion_res["dof_vel"],
    # )

    # rollouter.single_target_rollout(s_0, g, 200)
    # rollouter.motion_target_rollout(motion_lib, dt)
    # rollouter.motion_target_rollout(motion_lib, dt)
    s_0 = rollouter.random_action_rollout(s_0, 1)
    s_0 = rollouter.random_action_rollout(None, 1)
    s_0 = rollouter.random_action_rollout(None, 1)
    s_0 = rollouter.random_action_rollout(None, 1)
    s_0 = rollouter.random_action_rollout(None, 1)
    # env.close()


if __name__ == "__main__":
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)
    main()
