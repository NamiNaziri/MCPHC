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
            mcp: torch.nn.Module,
            pnn: torch.nn.Module,
            ae: torch.nn.Module,
            running_mean: torch.Tensor,
            running_var: torch.Tensor,
    ):
        self.env = env
        self.skeleton_trees = skeleton_trees
        self.mcp = mcp
        self.pnn = pnn
        self.ae = ae
        self.running_mean = running_mean
        self.running_var = running_var

    def single_target_rollout(self, s_0: IsaacS0, g: IsaacS0, n_steps: int):
        # Retrieve target body_pos and body_vel
        self.env.reset(None)
        self.env.task._humanoid_root_states[:] = g.root_states * 1
        self.env.task._dof_pos[:] = g.dof_pos * 1
        self.env.task._dof_vel[:] = g.dof_vel * 1
        self.env.task._reset_env_tensors(None)
        self.env.task._refresh_sim_tensors()
        self.env.task.gym.simulate(self.env.task.sim)
        self.env.task._reset_env_tensors(None)
        self.env.task._refresh_sim_tensors()

        ref_body_pos = self.env.task._rigid_body_pos[:] * 1
        ref_body_vel = self.env.task._rigid_body_vel[:] * 1

        # # Retrieve target body_pos via skeletonstate
        # pose_quat = sRot.from_euler("XYZ", g.dof_pos.reshape(-1, 3), degrees=False).as_quat().reshape(-1, 23, 4)
        # # pose_quat = sRot.from_rotvec(g.dof_pos.reshape(-1, 3)).as_quat().reshape(-1, 23, 4)
        # # pose_quat = np.concatenate([g.root_states[:, None, 3:7], pose_quat], axis=1)
        # new_sk_state = SkeletonState.from_rotation_and_root_translation(
        #     self.skeleton_trees[0],
        #     torch.as_tensor(pose_quat),
        #     g.root_states[:, :3].cpu(),
        #     is_local=True
        # )
        # ref_body_pos = new_sk_state.global_translation
        # ref_body_vel = self.env.task._rigid_body_vel[:] * 0
        # ref_body_pos = ref_body_pos.cuda()
        # ref_body_vel = ref_body_vel.cuda()

        # Then onto setting initial state
        self.env.reset(None)
        self.env.task._humanoid_root_states[:] = s_0.root_states
        self.env.task._dof_pos[:] = s_0.dof_pos
        self.env.task._dof_vel[:] = s_0.dof_vel
        self.env.task._reset_env_tensors(None)
        obs = self.env.task._compute_humanoid_obs(None)

        # Perform rollout
        for t in tqdm(range(n_steps)):
            body_pos = self.env.task._rigid_body_pos
            body_rot = self.env.task._rigid_body_rot
            body_vel = self.env.task._rigid_body_vel
            root_pos = body_pos[..., 0, :]
            root_rot = body_rot[..., 0, :]

            ref_rb_pos_subset = ref_body_pos * 1
            ref_body_vel_subset = ref_body_vel * 1

            close_distance = 0.25
            distance = torch.norm(root_pos - ref_rb_pos_subset[..., 0, :], dim=-1)

            zeros_subset = distance > close_distance
            ref_rb_pos_subset[zeros_subset, 1:] = body_pos[zeros_subset, 1:]
            ref_body_vel_subset[zeros_subset, :] = body_vel[zeros_subset, :]

            g = compute_imitation_observations_v7(root_pos, root_rot, body_pos, body_vel, ref_rb_pos_subset, ref_body_vel_subset, 1, True)
            nail = torch.cat([obs, g], axis=-1)
            nail = ((nail - self.running_mean[None].float()) / torch.sqrt(self.running_var[None].float() + 1e-05))
            nail = torch.clamp(nail, min=-5.0, max=5.0)

            # Compute the PNN policy's actions
            _, actions = self.pnn(nail)
            x_all = torch.stack(actions, dim=1)

            # Compute the MCP policy's actions
            input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": nail,
                "rnn_states": None,
            }

            with torch.no_grad():
                res_dict = self.mcp(input_dict)
            weights = res_dict["mus"] * 1
            rescaled_weights = rescale_actions(-1., 1., torch.clamp(weights, -1.0, 1.0))

            act = torch.sum(rescaled_weights[:, :, None] * x_all, dim=1)
            self.env.task._marker_pos[:] = ref_body_pos
            self.env.task.gym.set_actor_root_state_tensor_indexed(self.env.task.sim, gymtorch.unwrap_tensor(self.env.task._root_states), gymtorch.unwrap_tensor(self.env.task._marker_actor_ids), len(self.env.task._marker_actor_ids))

            self.env.step(act)
            next_obs = self.env.task._compute_humanoid_obs(None)
            obs = next_obs * 1

    def motion_target_rollout(self, motion_lib: MotionLibSMPL, dt: float):
        body_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
        mask_names = ['R_Shoulder', 'R_Elbow', 'R_Wrist']
        mask_idx = [body_names.index(q) for q in mask_names]
        hand_idx = body_names.index("R_Hand")

        motion_res = motion_lib.get_motion_state(torch.tensor([0], device=motion_lib._device), torch.tensor([0], device=motion_lib._device), offset=None)
        hand_pos = motion_res["rg_pos"][:, hand_idx]

        self.env.reset(None)
        self.env.task._humanoid_root_states[:] = torch.cat([motion_res["root_pos"], motion_res["root_rot"], motion_res["root_vel"], motion_res["root_ang_vel"]], dim=-1)
        self.env.task._dof_pos[:] = motion_res["dof_pos"]
        self.env.task._dof_vel[:] = motion_res["dof_vel"]
        self.env.task._reset_env_tensors(None)
        self.env.task._refresh_sim_tensors()
        obs = self.env.task._compute_humanoid_obs(None)

        motion_length = motion_lib.get_motion_length(0).item()
        for t in tqdm(range(int(motion_length / dt))):
            motion_res = motion_lib.get_motion_state(torch.tensor([0], device=motion_lib._device), torch.tensor([dt * t], device=motion_lib._device), offset=None)
            ref_body_pos = motion_res["rg_pos"]
            ref_body_pos[:, hand_idx] = hand_pos
            ref_body_vel = motion_res["body_vel"]
            ref_body_vel[:, hand_idx] = 0

            body_pos = self.env.task._rigid_body_pos
            body_rot = self.env.task._rigid_body_rot
            body_vel = self.env.task._rigid_body_vel
            root_pos = body_pos[..., 0, :]
            root_rot = body_rot[..., 0, :]

            ref_rb_pos_subset = ref_body_pos * 1
            ref_body_vel_subset = ref_body_vel * 1

            # Masking
            random_occlu_idx = torch.tensor(mask_idx)
            ref_rb_pos_subset[:, random_occlu_idx] = body_pos[:, random_occlu_idx]

            close_distance = 0.25
            distance = torch.norm(root_pos - ref_rb_pos_subset[..., 0, :], dim=-1)

            zeros_subset = distance > close_distance
            ref_rb_pos_subset[zeros_subset, 1:] = body_pos[zeros_subset, 1:]
            ref_body_vel_subset[zeros_subset, :] = body_vel[zeros_subset, :]

            g = compute_imitation_observations_v7(root_pos, root_rot, body_pos, body_vel, ref_rb_pos_subset, ref_body_vel_subset, 1, True)
            nail = torch.cat([obs, g], axis=-1)
            nail = ((nail - self.running_mean[None].float()) / torch.sqrt(self.running_var[None].float() + 1e-05))
            nail = torch.clamp(nail, min=-5.0, max=5.0)

            # Compute the PNN policy's actions
            _, actions = self.pnn(nail)
            x_all = torch.stack(actions, dim=1)

            # Compute the MCP policy's actions
            input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": nail,
                "rnn_states": None,
            }

            with torch.no_grad():
                res_dict = self.mcp(input_dict)
            weights = res_dict["mus"] * 1
            rescaled_weights = rescale_actions(-1., 1., torch.clamp(weights, -1.0, 1.0))

            act = torch.sum(rescaled_weights[:, :, None] * x_all, dim=1)
            ref_body_pos_to_visualize = ref_body_pos * 1
            ref_body_pos_to_visualize[:, random_occlu_idx] *= 0
            self.env.task._marker_pos[:] = ref_body_pos_to_visualize
            # self.env.task._marker_pos[:] = ref_rb_pos_subset
            self.env.task.gym.set_actor_root_state_tensor_indexed(self.env.task.sim, gymtorch.unwrap_tensor(self.env.task._root_states), gymtorch.unwrap_tensor(self.env.task._marker_actor_ids), len(self.env.task._marker_actor_ids))

            self.env.step(act)
            next_obs = self.env.task._compute_humanoid_obs(None)
            obs = next_obs * 1

    def ae_rollout(self, s_0: IsaacS0, n_steps: int):
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
        ref_body_pos = self.ae.decoder.forward(noise)
        ref_body_pos = ref_body_pos.reshape(ref_body_pos.shape[0], -1, 3)
        ref_body_vel = self.env.task._rigid_body_vel * 0

        # Perform rollout
        for t in tqdm(range(n_steps)):
            body_pos = self.env.task._rigid_body_pos
            body_rot = self.env.task._rigid_body_rot
            body_vel = self.env.task._rigid_body_vel
            root_pos = body_pos[..., 0, :]
            root_rot = body_rot[..., 0, :]

            ref_rb_pos_subset = ref_body_pos * 1
            ref_body_vel_subset = ref_body_vel * 1

            close_distance = 0.25
            distance = torch.norm(root_pos - ref_rb_pos_subset[..., 0, :], dim=-1)

            zeros_subset = distance > close_distance
            ref_rb_pos_subset[zeros_subset, 1:] = body_pos[zeros_subset, 1:]
            ref_body_vel_subset[zeros_subset, :] = body_vel[zeros_subset, :]

            g = compute_imitation_observations_v7(root_pos, root_rot, body_pos, body_vel, ref_rb_pos_subset, ref_body_vel_subset, 1, True)
            nail = torch.cat([obs, g], axis=-1)
            nail = ((nail - self.running_mean[None].float()) / torch.sqrt(self.running_var[None].float() + 1e-05))
            nail = torch.clamp(nail, min=-5.0, max=5.0)

            # Compute the PNN policy's actions
            with torch.no_grad():
                _, actions = self.pnn(nail)
            x_all = torch.stack(actions, dim=1)

            # Compute the MCP policy's actions
            input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": nail,
                "rnn_states": None,
            }

            with torch.no_grad():
                weights, _, _, _ = self.mcp(input_dict)
            # weights = res_dict["mus"] * 1
            rescaled_weights = rescale_actions(-1., 1., torch.clamp(weights, -1.0, 1.0))

            act = torch.sum(rescaled_weights[:, :, None] * x_all, dim=1)
            self.env.task._marker_pos[:] = ref_body_pos
            self.env.task.gym.set_actor_root_state_tensor_indexed(self.env.task.sim, gymtorch.unwrap_tensor(self.env.task._root_states), gymtorch.unwrap_tensor(self.env.task._marker_actor_ids), len(self.env.task._marker_actor_ids))

            self.env.step(act)
            next_obs = self.env.task._compute_humanoid_obs(None)
            obs = next_obs * 1

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
    args.task = "Humanoid"
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

    exec("from typing import Tuple")
    exec("print(Tuple)")

    ae_dict = torch.load("data/theirs/ae.pkl")
    for line in ae_dict['imports'].split("\n"):
        exec(line)
    exec(ae_dict['model_src'])
    ae = eval(ae_dict['model_cls_name'])(*ae_dict['model_args'], **ae_dict['model_kwargs'])
    ae.load_state_dict(ae_dict['model_state_dict'])
    ae.requires_grad_(False)
    ae = ae.to("cuda")

    pnn_d = torch.load("data/theirs/pnn.pkl")

    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    mlp_args = torch.load("data/theirs/mlp_args.pkl")
    state_dict_load = pnn_d["pnn_state_dict"]
    pnn = load_pnn(mlp_args, state_dict_load, 4, False, "relu", "cuda")

    running_mean = pnn_d["obs_running_mean"].cuda()
    running_var = pnn_d["obs_running_var"].cuda()

    acb = torch.load("data/theirs/their_AMPContinuous_build.pkl")
    net = acb["network_builder"]('amp', **acb["config"])
    for name, _ in net.named_parameters():
        print(name)
    mcp2 = ModelAMPContinuous.Network(net).a2c_network
    mcp2_state_dict = torch.load("data/theirs/mcp2_state_dict.pkl")
    mcp2.load_state_dict(mcp2_state_dict)
    mcp = mcp2
    mcp.cuda()
    cnt = 0
    dt = 1.0 / 60.0

    rollouter = Rollouter(env, motion_lib_args["skeleton_trees"], mcp, pnn, ae, running_mean, running_var)

    motion_lib.load_motions(random_sample=False, start_idx=1, max_len=-1, **motion_lib_args)
    motion_length = motion_lib.get_motion_length(0).item()

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
    s_0 = rollouter.ae_rollout(s_0, 100)
    s_0 = rollouter.ae_rollout(None, 100)
    s_0 = rollouter.ae_rollout(None, 100)
    s_0 = rollouter.ae_rollout(None, 100)
    s_0 = rollouter.ae_rollout(None, 100)
    # env.close()


if __name__ == "__main__":
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)
    main()
