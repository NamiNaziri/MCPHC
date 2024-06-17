# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from

#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uuid import uuid4
import numpy as np
import os
import PIL.Image
from torchvision.transforms import ToTensor
import multiprocessing
from scipy.spatial.transform import Rotation
import joblib
import time

import pauli
from isaacgymenvs.tasks.base.vec_task import VecTask

# from mlexp_utils.dirs import proj_dir
from fbpp.utils import proj_dir
from phc.learning.amp_models import ModelAMPContinuous
from phc.learning.pnn import PNN
from phc.utils import torch_utils
from uhc.smpllib.smpl_local_robot import SMPL_Robot

# from omni. import _debug_draw
from phc.utils.flags import flags
from phc.env.tasks.base_task import BaseTask
from tqdm import tqdm
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from collections import defaultdict
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import gc
from phc.utils.draw_utils import agt_color, get_color_gradient
from phc.utils.easy_plot import EasyPlot

from phc.utils.motion_lib_smpl import MotionLibSMPL
import re
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch

#### NOTE: VERY IMPORTANT NOTE #####
# The memory layout of the actors are very important, Currently, first all of the humanoid actors are instansiated, then markers and then boxes.
# Humanoid actors being first directly impacts the indexing in setup_tensor function. so if you make change to the way actors are created, you should
# also adjust the setup_tensor function.

# TODO: at the moment because of shape of humaniod root, yyou won't be able to run with one agent in multi envs
angle = 0.72 * np.pi
num_agents = 1
first_agent = 0
second_agent = 1
main_agent = first_agent
HACK_MOTION_SYNC = False
ENABLE_MAX_COORD_OBS = True
normal = False
latent_dim = 31
# PERTURB_OBJS = [
#     ["small", 60],
#     ["small", 7],
#     ["small", 10],
#     ["small", 35],
#     ["small", 2],
#     ["small", 2],
#     ["small", 3],
#     ["small", 2],
#     ["small", 2],
#     ["small", 3],
#     ["small", 2],
#     ["large", 60],
#     ["small", 300],
# ]
PERTURB_OBJS = [
    ["small", 60],
    # ["large", 60],
]
agent_pos = []


def load_pnn(
    mlp_args, state_dict_load, num_prim, has_lateral, activation="relu", device="cpu"
):
    net_key_name = "actors.0"
    loading_keys = [
        k
        for k in state_dict_load.keys()
        if k.startswith(net_key_name) and k.endswith("bias")
    ]
    layer_size = []
    for idx, key in enumerate(loading_keys):
        layer_size.append(state_dict_load[key].shape[::-1][0])

    pnn = PNN(
        mlp_args, output_size=layer_size[-1], numCols=num_prim, has_lateral=has_lateral
    )
    pnn.load_state_dict(state_dict_load)

    pnn.freeze_pnn(num_prim)
    pnn.to(device)
    return pnn


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


def sixd_to_rotmat(sixd: torch.Tensor) -> torch.Tensor:
    sixd_reshaped = sixd.reshape(-1, 3, 2)
    third_column = torch.cross(sixd_reshaped[..., 0], sixd_reshaped[..., 1], dim=-1)
    rotmat = torch.cat([sixd_reshaped, third_column[..., None]], dim=-1)
    return rotmat


def remove_root_yaw_from_sixd(sixd: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    rotmat = sixd_to_rotmat(sixd)
    rot = Rotation.from_matrix(rotmat.reshape(-1, 3, 3))
    euler = rot.as_euler("zyx").reshape((-1, 24, 3))
    root_yaw = euler[:, 0, 0]
    root_yaw_rep = root_yaw[..., None].repeat(24, -1).reshape(-1)
    correction = Rotation.from_euler("z", -root_yaw_rep)
    corrected_rot = correction * rot
    corrected_rotmat = corrected_rot.as_matrix().reshape((-1, 24, 3, 3))
    new_sixd = torch.as_tensor(
        corrected_rotmat[..., :2].reshape((-1, 24, 6)),
        dtype=sixd.dtype,
        device=sixd.device,
    )
    return new_sixd, root_yaw


def add_root_yaw_to_sixd(sixd: torch.Tensor, root_yaw: torch.Tensor) -> torch.Tensor:
    rotmat = sixd_to_rotmat(sixd)
    root_yaw_rep = root_yaw[..., None].repeat(24, -1).reshape(-1)
    rot = Rotation.from_matrix(rotmat.reshape(-1, 3, 3))
    correction = Rotation.from_euler("z", root_yaw_rep)
    corrected_rot = correction * rot
    corrected_rotmat = corrected_rot.as_matrix().reshape((-1, 24, 3, 3))
    new_sixd = corrected_rotmat[..., :2].reshape(-1, 24, 6)
    return new_sixd


class HumanoidAeMcpPnn6(VecTask):
    """
    A humanoid character following PHC character specification,
    controlled via latents for the input AE. Then the downstream actions are
    all computed via MCP-PNN.
    """

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        # def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        #plt.ion()  # Turn on interactive mode
        i = 4
        j = 3
        self.fig, self.axs = plt.subplots(i,j, figsize=(i * 15, j * 5))  # Creating two subplots horizontally
        plt.tight_layout()
        #plt.subplots_adjust(left=0.9, right=1, top=1, bottom=0.9)
        self.fig.suptitle('Dynamic Plots')  # Adding a title to the entire figure
        self.easy_plot = EasyPlot()
        self.cfg = cfg
        # self.cfg = cfg
        # self.sim_params = sim_params
        # self.physics_engine = physics_engine
        # self.has_task = False
        # self.num_agents = num_agents
        self.load_humanoid_configs(cfg)
        self.first_cam_update = True
        self.ae_type = self.cfg["env"]["ae_type"]
        self.actor_init_pos = self.cfg["env"]["actor_init_pos"]
        self.root_motion = self.cfg["env"]["root_motion"]
        self.physics_enable = self.cfg["env"]["physics_enable"]
        self.gate_pvae = self.cfg["env"]["gate_pvae"]
        self.calc_root_dir = self.cfg["env"]["calc_root_dir"]
        self.root_dir_action = self.cfg["env"]["root_dir_action"]
        self.sweep = self.cfg["env"]["sweep"]
        print(f'sweep: {self.sweep}')
        self.sweep1 = self.cfg["env"]["sweep1"]
        print(f'sweep1: {self.sweep1}')

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        self.temp_running_mean = self.cfg["env"].get("temp_running_mean", True)
        self.partial_running_mean = self.cfg["env"].get("partial_running_mean", False)
        self.self_obs_v = self.cfg["env"].get("self_obs_v", 1)

        self.key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(self.key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()
        # self.cfg["device_type"] = device_type
        # self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        # Porting stuff over from HumanoidIm
        self._num_joints = len(self._body_names)
        # self.device = "cpu"
        # self.device_type = cfg.get("device_type", "cuda")
        # self.device_id = cfg.get("device_id", 0)
        # if self.device_type == "cuda" or self.device_type == "GPU":
        #     self.device = "cuda" + ":" + str(self.device_id)
        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0
        self.device = "cpu"
        if self.cfg["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
        self._track_bodies = self.cfg["env"].get("trackBodies", self._full_track_bodies)
        self._track_bodies_id = self._build_key_body_ids_tensor(self._track_bodies)
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # super().__init__(cfg=self.cfg)

        # motion related stuff
        self._motion_start_times = torch.zeros([self.num_envs, num_agents]).to(
            self.device
        )
        self._sampled_motion_ids = torch.arange(self.num_envs * num_agents).to(
            self.device
        )
        self._sampled_motion_ids = self._sampled_motion_ids.reshape(
            self.num_envs, num_agents
        )
        # self._sampled_motion_ids = torch.zeros(self.num_envs).long().to(self.device)
        self.motion_file = "./data/amass/pkls/amass_copycat_take5_train_circle.pkl"  # TODO: cfg['env']['motion_file']
        self._load_motion(self.motion_file)
        self.ref_motion_cache = {}
        self._motion_start_times_offset = torch.zeros([self.num_envs, num_agents]).to(
            self.device
        )
        self._min_motion_len = cfg["env"].get("min_length", -1)
        self.seq_motions = cfg["env"].get("seq_motions", False)

        self._global_offset = torch.zeros([self.num_envs, num_agents, 3]).to(
            self.device
        )

        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * self.sim_params.dt
        self._setup_tensors()
        self.self_obs_buf = torch.zeros(
            (self.num_envs, self.get_self_obs_size()),
            device=self.device,
            dtype=torch.float,
        )
        self.reward_raw = torch.zeros((self.num_envs, 1)).to(self.device)

        # Porting stuff over from HumanoidIm
        self._build_marker_state_tensors()
        self._build_box_state_tensors()

        # Set up wrappers for pretrained networks
        # First: Autoencoder
        # ae_dict = torch.load(f"{proj_dir}/data/theirs/ae.pkl")
        if self.ae_type == "ae":
            ae_dict = torch.load(f"{proj_dir}/good/ae2.pkl")
        elif self.ae_type == "vae":
            ae_dict = torch.load(f"{proj_dir}/good/vae11.0.pkl")
        elif self.ae_type == "cvae":
            self.sk_tree = SkeletonTree.from_mjcf(
                f"{proj_dir}/data/mjcf/smpl_humanoid_1.xml"
            )
            ae_dict = torch.load(f"{proj_dir}/good/vae_000540.pkl")
        elif self.ae_type == "pvae":
            self.sk_tree = SkeletonTree.from_mjcf(
                f"{proj_dir}/data/mjcf/smpl_humanoid_1.xml"
            )
            #ae_dict = torch.load(f"{proj_dir}/good/vae_000605.pkl") pvae 5 parts
            #ae_dict = torch.load(f"{proj_dir}/good/vae_000140.pkl") #pvae 2 parts
            ae_dict = torch.load(f"{proj_dir}/good/vae_000100.pkl") #pvae 2 parts only circle anims
            #ae_dict = torch.load(f"{proj_dir}/good/vae_000060.pkl") #pvae 2 parts all anims but altred upper and lower body
        elif self.ae_type == "pvae_dof":
            self.sk_tree = SkeletonTree.from_mjcf(
                f"{proj_dir}/data/mjcf/smpl_humanoid_1.xml"
            )
            #ae_dict = torch.load(f"{proj_dir}/good/vae_000160.pkl") #pvae with dof for 2 parts, 1k anims
            #ae_dict = torch.load(f"{proj_dir}/good/vae_000300.pkl") #pvae with dof for 2 parts, only circle anim
            #ae_dict = torch.load(f"{proj_dir}/good/vae_000200.pkl") #pvae with dof for 2 parts, circle + 1k anim
            ae_dict = torch.load(f"{proj_dir}/good/vae_000180.pkl") #pvae with dof for 2 parts, circle + 1k anim, kld 1e-4, 32dim, 2048 hidden
            #ae_dict = torch.load(f"{proj_dir}/good/vae_001240.pkl") #pvae with dof for 2 parts, circle + 1k anim, kld 1e-2, 32dim, 2048 hidden
            #ae_dict = torch.load(f"{proj_dir}/good/vae_001940.pkl") #five_part_kld_1e-6_latent_32_hidden_2048
            

            
        else:
            self.sk_tree = SkeletonTree.from_mjcf(
                f"{proj_dir}/data/mjcf/smpl_humanoid_1.xml"
            )
            ae_dict = torch.load(f"{proj_dir}/good/vae_000540.pkl")

        ae = pauli.load(ae_dict)
        ae.load_state_dict(ae_dict["model_state_dict"])
        ae.requires_grad_(False)
        ae.eval()
        ae = ae.to(self.device)
        self.ae = ae

        # Next: MCP
        acb = torch.load("data/theirs/their_AMPContinuous_build.pkl")
        net = acb["network_builder"]("amp", **acb["config"])
        for name, _ in net.named_parameters():
            print(name)
        mcp = ModelAMPContinuous.Network(net).a2c_network
        mcp_state_dict = torch.load("data/theirs/mcp2_state_dict.pkl")
        mcp.load_state_dict(mcp_state_dict)
        mcp.requires_grad_(False)
        mcp.eval()
        mcp.to(self.device)
        self.mcp = mcp

        # Finally: PNN
        pnn_d = torch.load("data/theirs/pnn.pkl")
        mlp_args = torch.load("data/theirs/mlp_args.pkl")
        state_dict_load = pnn_d["pnn_state_dict"]
        pnn = load_pnn(mlp_args, state_dict_load, 4, False, "relu", "cuda")
        pnn.requires_grad_(False)
        pnn.eval()
        pnn.to(self.device)
        self.pnn = pnn

        self.running_mean = pnn_d["obs_running_mean"].to(self.device)
        self.running_var = pnn_d["obs_running_var"].to(self.device)

        self.hlc_control_freq_inv = 1
        self.rew_hist = torch.zeros(
            (self.num_envs, self.hlc_control_freq_inv),
            device=self.device,
            dtype=torch.float,
        )
        self.blue_rb_xyz = None
        return

    def _load_motion(self, motion_file):
        assert self._dof_offsets[-1] == self.num_dof
        if self.smpl_humanoid:
            self._motion_lib = MotionLibSMPL(
                motion_file=motion_file,
                device=self.device,
                masterfoot_conifg=self._masterfoot_config,
            )
            motion = self._motion_lib.load_motions(
                skeleton_trees=self.skeleton_trees,
                gender_betas=self.humanoid_shapes.cpu(),
                limb_weights=self.humanoid_limb_and_weights.cpu(),
                random_sample=not HACK_MOTION_SYNC,
            )

        else:
            self._motion_lib = MotionLib(
                motion_file=motion_file,
                dof_body_ids=self._dof_body_ids,
                dof_offsets=self._dof_offsets,
                key_body_ids=self._key_body_ids.cpu().numpy(),
                device=self.device,
            )

        return

    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
        ## Cache the motion + offset
        if (
            offset is None
            or not "motion_ids" in self.ref_motion_cache
            or self.ref_motion_cache["offset"] is None
            or len(self.ref_motion_cache["motion_ids"]) != len(motion_ids)
            or len(self.ref_motion_cache["offset"]) != len(offset)
            or (self.ref_motion_cache["motion_ids"] - motion_ids).abs().sum()
            + (self.ref_motion_cache["motion_times"] - motion_times).abs().sum()
            + (self.ref_motion_cache["offset"] - offset).abs().sum()
            > 0
        ):
            self.ref_motion_cache["motion_ids"] = (
                motion_ids.clone()
            )  # need to clone; otherwise will be overriden
            self.ref_motion_cache["motion_times"] = (
                motion_times.clone()
            )  # need to clone; otherwise will be overriden
            self.ref_motion_cache["offset"] = (
                offset.clone() if not offset is None else None
            )
        else:
            # print('using cache')
            return self.ref_motion_cache
        # print('new motion res')
        motion_res = self._motion_lib.get_motion_state(
            motion_ids, motion_times, offset=offset
        )
        # motion_res["rg_pos"][:, :, :2]  -=  motion_res["rg_pos"][:,[0],:2]
        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache

    def _load_proj_asset(self):
        asset_root = "src/phc/data/assets/mjcf/"

        # small_asset_file = "block_projectile.urdf"
        small_asset_file = "hit_target.urdf"
        # small_asset_file = "ball_medium.urdf"
        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 100.0
        # small_asset_options.fix_base_link = True
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._small_proj_asset = self.gym.load_asset(
            self.sim, asset_root, small_asset_file, small_asset_options
        )

        large_asset_file = "block_projectile_large.urdf"
        large_asset_options = gymapi.AssetOptions()
        large_asset_options.angular_damping = 0.01
        large_asset_options.linear_damping = 0.01
        large_asset_options.max_angular_velocity = 100.0
        large_asset_options.density = 10000000.0
        # large_asset_options.fix_base_link = True
        large_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._large_proj_asset = self.gym.load_asset(
            self.sim, asset_root, large_asset_file, large_asset_options
        )
        return

    def _build_proj(self, env_id, env_ptr, pos_add):
        pos = [
            # [-0.01, 0.0, 0.0],
            [2.5, 0.3051, 0.0]
            # [ 0.0890016, -0.40830246, 0.25]
        ]
        for i, obj in enumerate(PERTURB_OBJS):
            default_pose = gymapi.Transform()
            torch.manual_seed(int(time.time()))
            # default_pose.p.x = pos[i][0] + torch.rand(1) * 1.2 + pos_add
            # default_pose.p.y = pos[i][1] + torch.rand(1) * 1.2
            # default_pose.p.z = pos[i][2]
            default_pose.p.x = pos[i][0]
            default_pose.p.y = pos[i][1]
            default_pose.p.z = pos[i][2]
            obj_type = obj[0]
            if obj_type == "small":
                proj_asset = self._small_proj_asset
            elif obj_type == "large":
                proj_asset = self._large_proj_asset

            proj_handle = self.gym.create_actor(
                env_ptr, proj_asset, default_pose, "proj{:d}".format(i), env_id, 2
            )
            # Set collision filter so only the sword will hit it
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, proj_handle)
            props[0].filter = 268435455
            self.gym.set_actor_rigid_shape_properties(env_ptr, proj_handle, props)
            self._proj_handles.append(proj_handle)

        return

    def _setup_tensors(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        # ZL: needs to put this back
        # sensors_per_env = 2
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        # dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        # self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,num_agents, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()

        self._root_states_reshaped = self._root_states.view(
            self.num_envs, num_actors, actor_root_state.shape[-1]
        )
        self._humanoid_root_states = self._root_states_reshaped[..., :num_agents, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[..., 7:13] = 0
        self._initial_humanoid_root_states[..., 2] += 0.1

        #NOTE num_actor could be wrong! this is working because we are doing this before bulding markers and the box. I think it should be num_agents instead
        self._humanoid_actor_ids = num_actors * torch.arange(
            self.num_envs, device=self.device, dtype=torch.int32
        )

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs

        # TODO: ask if dof_state only used for humanoids!? if not this should be changed to use the prev solution but for agent lists

        # self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., main_agent * self.num_dof: main_agent * self.num_dof + self.num_dof, 0]
        # self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., main_agent * self.num_dof:main_agent * self.num_dof+ self.num_dof, 1]
        self._dof_pos = self._dof_state.view(
            self.num_envs, num_agents, self.num_dof, 2
        )[..., 0]
        self._dof_vel = self._dof_state.view(
            self.num_envs, num_agents, self.num_dof, 2
        )[..., 1]

        self._initial_dof_pos = torch.zeros_like(
            self._dof_pos, device=self.device, dtype=torch.float
        )
        self._initial_dof_vel = torch.zeros_like(
            self._dof_vel, device=self.device, dtype=torch.float
        )

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._initial_rigid_body_state = self._rigid_body_state.clone()
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(
            self.num_envs, bodies_per_env, 13
        )  # shape [num_env, num_bodies * num_agents + _num_joints * num_markers(should be equalt to num_agents) + num_boxes, 13]
        self._humanoid_rigid_body_state_reshaped = self._rigid_body_state_reshaped[
            ..., : num_agents * self.num_bodies, :
        ].view(self.num_envs, num_agents, self.num_bodies, 13)
        # self._num_joints is used for red dots #+1 is for the box acotr. if we have more boex should be more

        self._rigid_body_pos = self._humanoid_rigid_body_state_reshaped[..., 0:3]
        self._initial_rigid_body_pos = self._rigid_body_pos.clone()
        self._rigid_body_rot = self._humanoid_rigid_body_state_reshaped[..., 3:7]
        self._rigid_body_vel = self._humanoid_rigid_body_state_reshaped[..., 7:10]
        self._rigid_body_ang_vel = self._humanoid_rigid_body_state_reshaped[..., 10:13]

        self._initial_box_states = self._root_states_reshaped[:, -1].clone()

        if self.self_obs_v == 2:
            self._rigid_body_pos_hist = torch.zeros(
                (self.num_envs, self.past_track_steps, self.num_bodies, 3),
                device=self.device,
                dtype=torch.float,
            )
            self._rigid_body_rot_hist = torch.zeros(
                (self.num_envs, self.past_track_steps, self.num_bodies, 4),
                device=self.device,
                dtype=torch.float,
            )
            self._rigid_body_vel_hist = torch.zeros(
                (self.num_envs, self.past_track_steps, self.num_bodies, 3),
                device=self.device,
                dtype=torch.float,
            )
            self._rigid_body_ang_vel_hist = torch.zeros(
                (self.num_envs, self.past_track_steps, self.num_bodies, 3),
                device=self.device,
                dtype=torch.float,
            )

        # TODO: maybe this also has to be changed for each character DONE: Did this!
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., : self.num_bodies * num_agents, :]

        self._terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long
        )

        self._build_termination_heights()

        # populate for all of the agents
        self._termination_heights = self._termination_heights.repeat(num_agents)
        contact_bodies = self.cfg["env"]["contactBodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(self.key_bodies)

        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)
        # populate for all of the agents
        self._contact_body_ids = torch.cat(
            [self._contact_body_ids + self.num_bodies * ii for ii in range(num_agents)]
        )

        if self.viewer != None:
            self._init_camera()

    def load_humanoid_configs(self, cfg):
        # self.smpl_humanoid = cfg["env"]["asset"]['assetFileName'] == "mjcf/smpl_humanoid.xml"
        self.smpl_humanoid = True
        if self.smpl_humanoid:
            self.load_smpl_configs(cfg)

    def load_common_humanoid_configs(self, cfg):
        self._bias_offset = cfg["env"].get("bias_offset", False)
        self._divide_group = cfg["env"].get("divide_group", False)
        self._group_obs = cfg["env"].get("group_obs", False)
        self._disable_group_obs = cfg["env"].get("disable_group_obs", False)
        if self._divide_group:
            self._group_num_people = group_num_people = min(
                cfg["env"].get("num_env_group", 128), cfg["env"]["numEnvs"]
            )
            self._group_ids = torch.tensor(
                np.arange(cfg["env"]["numEnvs"] / group_num_people)
                .repeat(group_num_people)
                .astype(int)
            )

        self._has_shape_obs = cfg["env"].get("has_shape_obs", False)
        self._has_shape_obs_disc = cfg["env"].get("has_shape_obs_disc", False)
        self._has_limb_weight_obs = cfg["env"].get("has_weight_obs", False)
        self._has_limb_weight_obs_disc = cfg["env"].get("has_weight_obs_disc", False)
        self.has_shape_variation = cfg["env"].get("has_shape_variation", False)

        self._has_self_collision = cfg["env"].get("has_self_collision", False)
        self._has_mesh = cfg["env"].get("has_mesh", True)
        self._replace_feet = cfg["env"].get("replace_feet", True)  # replace feet or not
        self._has_jt_limit = cfg["env"].get("has_jt_limit", True)
        self._has_dof_subset = cfg["env"].get("has_dof_subset", False)
        self._has_smpl_pd_offset = cfg["env"].get("has_smpl_pd_offset", False)
        self.shape_resampling_interval = cfg["env"].get(
            "shape_resampling_interval", 100
        )
        self._remove_ankles = cfg["env"].get("remove_ankles", False)
        self._remove_neck = cfg["env"].get("remove_neck", False)
        self.getup_schedule = cfg["env"].get("getup_schedule", False)
        self._kp_scale = cfg["env"].get("kp_scale", 1.0)
        self._kd_scale = cfg["env"].get("kd_scale", self._kp_scale)
        self._freeze_toe = cfg["env"].get("freeze_toe", True)
        self.remove_toe_im = cfg["env"].get("remove_toe_im", True)
        self.hard_negative = cfg["env"].get(
            "hard_negative", False
        )  # hard negative sampling for im
        self.cycle_motion = cfg["env"].get(
            "cycle_motion", False
        )  # Cycle motion to reach 300
        self.power_reward = cfg["env"].get("power_reward", False)
        self.obs_v = cfg["env"].get("obs_v", 1)
        self.amp_obs_v = cfg["env"].get("amp_obs_v", 1)
        self._masterfoot = cfg["env"].get("masterfoot", False)

        ## Kin stuff
        self.kin_loss = cfg["env"].get("kin_loss", False)
        self.kin_lr = cfg["env"].get("kin_lr", 5e-4)
        self.z_readout = cfg["env"].get("z_readout", False)
        self.z_read = cfg["env"].get("z_read", False)
        self.z_uniform = cfg["env"].get("z_uniform", False)
        self.z_model = cfg["env"].get("z_model", False)
        self.distill = cfg["env"].get("distill", False)

        self.remove_disc_rot = cfg["env"].get("remove_disc_rot", False)

        ## ZL Devs
        #################### Devs ####################
        self.fitting = cfg["env"].get("fitting", False)
        self.zero_out_far = cfg["env"].get("zero_out_far", False)
        self.zero_out_far_train = cfg["env"].get("zero_out_far_train", True)
        self.max_len = cfg["env"].get("max_len", -1)
        self.cycle_motion_xp = cfg["env"].get(
            "cycle_motion_xp", False
        )  # Cycle motion, but cycle farrrrr.
        self.models_path = cfg["env"].get(
            "models",
            [
                "output/dgx/smpl_im_fit_3_1/Humanoid_00185000.pth",
                "output/dgx/smpl_im_fit_3_2/Humanoid_00198750.pth",
            ],
        )

        self.eval_full = cfg["env"].get("eval_full", False)
        self.auto_pmcp = cfg["env"].get("auto_pmcp", False)
        self.auto_pmcp_soft = cfg["env"].get("auto_pmcp_soft", False)
        self.strict_eval = cfg["env"].get("strict_eval", False)
        self.add_obs_noise = cfg["env"].get("add_obs_noise", False)

        self._occl_training = cfg["env"].get(
            "occl_training", False
        )  # Cycle motion, but cycle farrrrr.
        self._occl_training_prob = cfg["env"].get(
            "occl_training_prob", 0.1
        )  # Cycle motion, but cycle farrrrr.
        self._sim_occlu = False
        self._res_action = cfg["env"].get("res_action", False)
        self.close_distance = cfg["env"].get("close_distance", 0.25)
        self.far_distance = cfg["env"].get("far_distance", 3)
        self._zero_out_far_steps = cfg["env"].get("zero_out_far_steps", 90)
        self.past_track_steps = cfg["env"].get("past_track_steps", 5)
        #################### Devs ####################

    def load_smpl_configs(self, cfg):
        self.load_common_humanoid_configs(cfg)

        self._has_upright_start = cfg["env"].get("has_upright_start", True)
        self.remove_toe = cfg["env"].get("remove_toe", False)
        self.big_ankle = cfg["env"].get("big_ankle", False)
        self._remove_thorax = cfg["env"].get("remove_thorax", False)
        self._real_weight_porpotion_capsules = cfg["env"].get(
            "real_weight_porpotion_capsules", False
        )
        self._real_weight = cfg["env"].get("real_weight", False)

        self._master_range = cfg["env"].get("master_range", 30)
        self._freeze_toe = cfg["env"].get("freeze_toe", True)
        self._freeze_hand = cfg["env"].get("freeze_hand", True)
        self._box_body = cfg["env"].get("box_body", False)

        self.reduce_action = cfg["env"].get("reduce_action", False)
        if self._masterfoot:
            self.action_idx = [
                0,
                1,
                2,
                4,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                25,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                54,
                55,
                60,
                61,
                62,
                65,
                66,
                67,
                68,
                75,
                76,
                77,
                80,
                81,
                82,
                83,
            ]
        else:
            self.action_idx = [
                0,
                1,
                2,
                4,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                16,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                36,
                37,
                42,
                43,
                44,
                47,
                48,
                49,
                50,
                57,
                58,
                59,
                62,
                63,
                64,
                65,
            ]

        disc_idxes = []
        self._body_names_orig = [
            "Pelvis",
            "L_Hip",
            "L_Knee",
            "L_Ankle",
            "L_Toe",
            "R_Hip",
            "R_Knee",
            "R_Ankle",
            "R_Toe",
            "Torso",
            "Spine",
            "Chest",
            "Neck",
            "Head",
            "L_Thorax",
            "L_Shoulder",
            "L_Elbow",
            "L_Wrist",
            "L_Hand",
            "R_Thorax",
            "R_Shoulder",
            "R_Elbow",
            "R_Wrist",
            "R_Hand",
        ]
        _body_names_orig_copy = self._body_names_orig.copy()
        if self.remove_toe_im:
            _body_names_orig_copy.remove("L_Toe")
            _body_names_orig_copy.remove("R_Toe")
        self._full_track_bodies = _body_names_orig_copy

        _body_names_orig_copy = self._body_names_orig.copy()
        _body_names_orig_copy.remove(
            "L_Toe"
        )  # Following UHC as hand and toes does not have realiable data.
        _body_names_orig_copy.remove("R_Toe")
        _body_names_orig_copy.remove("L_Hand")
        _body_names_orig_copy.remove("R_Hand")
        self._eval_bodies = _body_names_orig_copy  # default eval bodies

        if self._masterfoot:
            self._body_names = [
                "Pelvis",
                "L_Hip",
                "L_Knee",
                "L_Ankle",
                "L_Toe",
                "L_Toe_1",
                "L_Toe_1_1",
                "L_Toe_2",
                "R_Hip",
                "R_Knee",
                "R_Ankle",
                "R_Toe",
                "R_Toe_1",
                "R_Toe_1_1",
                "R_Toe_2",
                "Torso",
                "Spine",
                "Chest",
                "Neck",
                "Head",
                "L_Thorax",
                "L_Shoulder",
                "L_Elbow",
                "L_Wrist",
                "L_Hand",
                "R_Thorax",
                "R_Shoulder",
                "R_Elbow",
                "R_Wrist",
                "R_Hand",
            ]
            self._body_to_orig = [
                self._body_names.index(name) for name in self._body_names_orig
            ]
            self._body_to_orig_without_toe = [
                self._body_names.index(name)
                for name in self._body_names_orig
                if name not in ["L_Toe", "R_Toe"]
            ]
            self.orig_to_orig_without_toe = [
                self._body_names_orig.index(name)
                for name in self._body_names_orig
                if name not in ["L_Toe", "R_Toe"]
            ]

            self._masterfoot_config = {
                "body_names_orig": self._body_names_orig,
                "body_names": self._body_names,
                "body_to_orig": self._body_to_orig,
                "body_to_orig_without_toe": self._body_to_orig_without_toe,
                "orig_to_orig_without_toe": self.orig_to_orig_without_toe,
            }
        else:
            self._body_names = self._body_names_orig
            self._masterfoot_config = None

        self._dof_names = self._body_names[1:]

        remove_names = ["L_Hand", "R_Hand", "L_Toe", "R_Toe"]

        if self._remove_ankles:
            remove_names.append("L_Ankle")
            remove_names.append("R_Ankle")
        if self._remove_thorax:
            remove_names.append("L_Thorax")
            remove_names.append("R_Thorax")
        if self._remove_neck:
            remove_names.append("Neck")
            remove_names.append("Head")

        if self.remove_disc_rot:
            remove_names = self._body_names_orig  # NO AMP Rotation

        if self._masterfoot:
            remove_names += [
                "L_Toe_1",
                "L_Toe_1_1",
                "L_Toe_2",
                "R_Toe_1",
                "R_Toe_1_1",
                "R_Toe_2",
            ]

        if self._masterfoot:
            self.limb_weight_group = [
                [
                    "L_Hip",
                    "L_Knee",
                    "L_Ankle",
                    "L_Toe",
                    "L_Toe_1",
                    "L_Toe_1_1",
                    "L_Toe_2",
                ],
                [
                    "R_Hip",
                    "R_Knee",
                    "R_Ankle",
                    "R_Toe",
                    "R_Toe_1",
                    "R_Toe_1_1",
                    "R_Toe_2",
                ],
                ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"],
                ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"],
                [
                    "Pelvis",
                    "Torso",
                    "Spine",
                    "Chest",
                    "Neck",
                    "Head",
                ],
            ]
            self.limb_weight_group = [
                [self._body_names.index(g) for g in group]
                for group in self.limb_weight_group
            ]
        else:
            self.limb_weight_group = [
                ["L_Hip", "L_Knee", "L_Ankle", "L_Toe"],
                ["R_Hip", "R_Knee", "R_Ankle", "R_Toe"],
                ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"],
                ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"],
                ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"],
            ]
            self.limb_weight_group = [
                [self._body_names.index(g) for g in group]
                for group in self.limb_weight_group
            ]

        for idx, name in enumerate(self._dof_names):
            if not name in remove_names:
                disc_idxes.append(np.arange(idx * 3, (idx + 1) * 3))

        self.dof_subset = (
            torch.from_numpy(np.concatenate(disc_idxes))
            if len(disc_idxes) > 0
            else torch.tensor([]).long()
        )
        self.left_indexes = [
            idx for idx, name in enumerate(self._dof_names) if name.startswith("L")
        ]
        self.right_indexes = [
            idx for idx, name in enumerate(self._dof_names) if name.startswith("R")
        ]

        self.left_lower_indexes = [
            idx
            for idx, name in enumerate(self._dof_names)
            if name.startswith("L") and name[2:] in ["Hip", "Knee", "Ankle", "Toe"]
        ]
        self.right_lower_indexes = [
            idx
            for idx, name in enumerate(self._dof_names)
            if name.startswith("R") and name[2:] in ["Hip", "Knee", "Ankle", "Toe"]
        ]

        self.selected_group_jts_names = [
            "Pelvis",
            "L_Hip",
            "R_Hip",
            "Torso",
            "L_Ankle",
            "R_Ankle",
            "L_Elbow",
            "R_Elbow",
            "L_Hand",
            "R_Hand",
        ]
        self.selected_group_jts = torch.tensor(
            [
                self._body_names.index(jt_name)
                for jt_name in self.selected_group_jts_names
            ]
        )

        if self._masterfoot:
            self.left_to_right_index = [
                0,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                15,
                16,
                17,
                18,
                19,
                25,
                26,
                27,
                28,
                29,
                20,
                21,
                22,
                23,
                24,
            ]
            self.left_to_right_index_action = [
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                14,
                15,
                16,
                17,
                18,
                24,
                25,
                26,
                27,
                28,
                19,
                20,
                21,
                22,
                23,
            ]
        else:
            self.left_to_right_index = [
                0,
                5,
                6,
                7,
                8,
                1,
                2,
                3,
                4,
                9,
                10,
                11,
                12,
                13,
                19,
                20,
                21,
                22,
                23,
                14,
                15,
                16,
                17,
                18,
            ]
            self.left_to_right_index_action = [
                4,
                5,
                6,
                7,
                0,
                1,
                2,
                3,
                8,
                9,
                10,
                11,
                12,
                18,
                19,
                20,
                21,
                22,
                13,
                14,
                15,
                16,
                17,
            ]

        self._load_amass_gender_betas()

        contact_bodies_names = [
            "Pelvis",
            "L_Hip",
            "L_Knee",
            "R_Hip",
            "R_Knee",
            "Torso",
            "Spine",
            "Chest",
            "Neck",
            "Head",
            "L_Thorax",
            "L_Shoulder",
            "L_Elbow",
            "L_Wrist",
            "L_Hand",
            "R_Thorax",
            "R_Shoulder",
            "R_Elbow",
            "R_Wrist",
            "R_Hand",
        ]
        self.contact_bodies_index = [
            self._body_names.index(body_name) for body_name in contact_bodies_names
        ]

    def _clear_recorded_states(self):
        del self.state_record
        self.state_record = defaultdict(list)

    def _record_states(self):
        self.state_record["dof_pos"].append(self._dof_pos.cpu().clone())
        self.state_record["root_states"].append(
            self._humanoid_root_states.cpu().clone()
        )
        self.state_record["progress"].append(self.progress_buf.cpu().clone())

    def _write_states_to_file(self, file_name):
        self.state_record["skeleton_trees"] = self.skeleton_trees
        self.state_record["humanoid_betas"] = self.humanoid_shapes
        print(f"Dumping states into {file_name}")

        progress = torch.stack(self.state_record["progress"], dim=1)
        progress_diff = torch.cat(
            [progress, -10 * torch.ones(progress.shape[0], 1).to(progress)], dim=-1
        )

        diff = torch.abs(progress_diff[:, :-1] - progress_diff[:, 1:])
        split_idx = torch.nonzero(diff > 1)
        split_idx[:, 1] += 1
        dof_pos_all = torch.stack(self.state_record["dof_pos"])
        root_states_all = torch.stack(self.state_record["root_states"])
        fps = 60
        motion_dict_dump = {}
        num_for_this_humanoid = 0
        curr_humanoid_index = 0

        for idx in range(len(split_idx)):
            split_info = split_idx[idx]
            humanoid_index = split_info[0]

            if humanoid_index != curr_humanoid_index:
                num_for_this_humanoid = 0
                curr_humanoid_index = humanoid_index

            if num_for_this_humanoid == 0:
                start = 0
            else:
                start = split_idx[idx - 1][-1]

            end = split_idx[idx][-1]

            dof_pos_seg = dof_pos_all[start:end, humanoid_index]
            B, H = dof_pos_seg.shape

            root_states_seg = root_states_all[start:end, humanoid_index]
            body_quat = torch.cat(
                [
                    root_states_seg[:, None, 3:7],
                    torch_utils.exp_map_to_quat(dof_pos_seg.reshape(B, -1, 3)),
                ],
                dim=1,
            )

            motion_dump = {
                "skeleton_tree": self.state_record["skeleton_trees"][
                    humanoid_index
                ].to_dict(),
                "body_quat": body_quat,
                "trans": root_states_seg[:, :3],
                "root_states_seg": root_states_seg,
                "dof_pos": dof_pos_seg,
            }
            motion_dump["fps"] = fps
            motion_dump["betas"] = (
                self.humanoid_shapes[humanoid_index].detach().cpu().numpy()
            )
            motion_dict_dump[f"{humanoid_index}_{num_for_this_humanoid}"] = motion_dump
            num_for_this_humanoid += 1

        joblib.dump(motion_dict_dump, file_name)
        self.state_record = defaultdict(list)

    def get_obs_size(self):
        return self.get_self_obs_size()

    def get_running_mean_size(self):
        return (self.get_obs_size(),)

    def get_self_obs_size(self):
        if self.self_obs_v == 1:
            return self._num_self_obs
        elif self.self_obs_v == 2:
            return self._num_self_obs * (self.past_track_steps + 1)

    def get_action_size(self):
        if self.ae_type == "cvae":

            # return (7 + 2 + 16) * num_agents  # for ys + latent space dim + root xy
            return (
                3 + 3 + 16
            ) * num_agents  # root xyz edit + root rpy edit + latent space edit

            # return 3
        elif self.ae_type == "pvae":
            actions = latent_dim
            if(self.gate_pvae):
                actions += 5
            if(self.root_motion == False):
                actions += 2 #root_xy
            return (
               actions
            ) * num_agents  # root xyz edit + root rpy edit + latent space edit
        elif self.ae_type == "pvae_dof":
            actions = latent_dim
            if(self.gate_pvae):
                actions += 5
            if(self.root_motion == False):
                actions += 2 #root_xy
            if(self.root_dir_action):
                actions +=1
            return (
               actions
            ) * num_agents  # root xyz edit + root rpy edit + latent space edit
        elif self.ae_type == "dof":
            return (
                69 + 2 #dof + root_xy
            ) * num_agents
        else:
            return 10 * num_agents
        # return self._num_actions

    def get_dof_action_size(self):
        return self._dof_size

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        # self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )
        return

    def reset(self, env_ids=None):
        safe_reset = (env_ids is None) or len(env_ids) == self.num_envs
        if env_ids is None:
            env_ids = to_torch(
                np.arange(self.num_envs), device=self.device, dtype=torch.long
            )

        self._cache_anim_root(env_ids)
        self.reset_idx(env_ids)
        self._reset_env_tensors(env_ids)

        # if safe_reset:
        #     # import ipdb; ipdb.set_trace()
        #     # print("3resetting here!!!!", self._humanoid_root_states[0, :3] - self._rigid_body_pos[0, 0])
        #     # ZL: This way it will simuate one step, then get reset again, squashing any remaining wiredness. Temporary fix
        #     self.gym.simulate(self.sim)
        #     self.reset_idx(env_ids)
        #     torch.cuda.empty_cache()

        # TODO: fix the function to use env_ids
        self._compute_observations(
            self._rigid_body_pos.reshape(
                self.num_envs * num_agents, self.num_bodies, 3
            ),
            self._rigid_body_rot.reshape(
                self.num_envs * num_agents,
                self.num_bodies,
                self._rigid_body_rot.shape[-1],
            ),
            self._rigid_body_vel.reshape(
                self.num_envs * num_agents,
                self.num_bodies,
                self._rigid_body_vel.shape[-1],
            ),
            self._rigid_body_ang_vel.reshape(
                self.num_envs * num_agents,
                self.num_bodies,
                self._rigid_body_ang_vel.shape[-1],
            ),
        )
        obs = self.obs_buf
        return obs[:]

    def change_char_color(self):
        colors = []
        offset = np.random.randint(0, 10)
        for env_id in range(self.num_envs):
            rand_cols = agt_color(env_id + offset)
            colors.append(rand_cols)

        self.sample_char_color(torch.tensor(colors), torch.arange(self.num_envs))

    def sample_char_color(self, cols, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr,
                    handle,
                    j,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(cols[env_id, 0], cols[env_id, 1], cols[env_id, 2]),
                )
        return

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr,
                    handle,
                    j,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(col[0], col[1], col[2]),
                )

        return

    def reset_idx(self, env_ids):

        if len(env_ids) > 0:

            self._reset_actors(
                env_ids
            )  # this funciton calle _set_env_state, and should set all state vectors
            self._reset_env_tensors(env_ids)

            self._refresh_sim_tensors()
            # self._rigid_body_state[env_ids] = self._initial_rigid_body_state[env_ids]
            if(self.physics_enable):
                self.gym.simulate(self.sim)

            # self._reset_actors(env_ids)  # this funciton calle _set_env_state, and should set all state vectors
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()

            if self.self_obs_v == 2:
                self._init_tensor_history(env_ids)

            # TODO: fix the function to use env_ids
            self._compute_observations(
                self._rigid_body_pos.reshape(
                    self.num_envs * num_agents, self.num_bodies, 3
                ),
                self._rigid_body_rot.reshape(
                    self.num_envs * num_agents,
                    self.num_bodies,
                    self._rigid_body_rot.shape[-1],
                ),
                self._rigid_body_vel.reshape(
                    self.num_envs * num_agents,
                    self.num_bodies,
                    self._rigid_body_vel.shape[-1],
                ),
                self._rigid_body_ang_vel.reshape(
                    self.num_envs * num_agents,
                    self.num_bodies,
                    self._rigid_body_ang_vel.shape[-1],
                ),
            )
            torch.cuda.empty_cache()

        return

    def _reset_env_tensors(self, env_ids):

        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        new_env_ids_int32 = torch.cat(
            [
                torch.arange(
                    start, start + num_agents, device=self.device, dtype=torch.int32
                )
                for start in env_ids_int32
            ]
        )
        # env_ids_int32 = torch.arange(start=env_ids_int32, end=env_ids_int32 + num_agents , device=self.device, dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(new_env_ids_int32),
            len(new_env_ids_int32),
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._box_actor_ids),
            len(self._box_actor_ids),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(new_env_ids_int32),
            len(new_env_ids_int32),
        )

        # for env_ptr, humanoid_ptr in zip(self.envs, self.humanoid_handles):
        #     rbs = self.gym.get_actor_rigid_body_states(env_ptr, humanoid_ptr, gymapi.STATE_ALL)
        #     self.gym.set_actor_rigid_body_states(self.gym, env_ptr, humanoid_ptr, gymtorch.unwrap_tensor(self._rigid_body_state), self.num_bodies)

        # print("#################### refreshing ####################")
        # print("rb", (self._rigid_body_state_reshaped[None, :] - self._rigid_body_state_reshaped[:, None]).abs().sum())
        # print("contact", (self._contact_forces[None, :] - self._contact_forces[:, None]).abs().sum())
        # print('dof_pos', (self._dof_pos[None, :] - self._dof_pos[:, None]).abs().sum())
        # print("dof_vel", (self._dof_vel[None, :] - self._dof_vel[:, None]).abs().sum())
        # print("root_states", (self._humanoid_root_states[None, :] - self._humanoid_root_states[:, None]).abs().sum())
        # print("#################### refreshing ####################")

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self._contact_forces[env_ids] = 0

        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction

        # plane_params.static_friction = 50
        # plane_params.dynamic_friction = 50

        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if asset_file == "mjcf/amp_humanoid.xml":
            ### ZL: changes
            self._dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
            self._dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
            self._dof_obs_size = 72
            self._num_actions = 28

            if ENABLE_MAX_COORD_OBS:
                self._num_self_obs = 1 + 15 * (3 + 6 + 3 + 3) - 3
            else:
                self._num_self_obs = (
                    13 + self._dof_obs_size + 28 + 3 * num_key_bodies
                )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

        elif (
            asset_file == "mjcf/smpl_humanoid.xml"
            or asset_file == "mjcf/smpl_humanoid2.xml"
        ):
            # import ipdb; ipdb.set_trace()
            self._dof_body_ids = np.arange(1, len(self._body_names))
            self._dof_offsets = np.linspace(
                0, len(self._dof_names) * 3, len(self._body_names)
            ).astype(int)
            self._dof_obs_size = len(self._dof_names) * 6
            self._dof_size = len(self._dof_names) * 3
            if self.reduce_action:
                self._num_actions = len(self.action_idx)
            else:
                self._num_actions = len(self._dof_names) * 3

            if ENABLE_MAX_COORD_OBS:
                self._num_self_obs = (
                    1 + len(self._body_names) * (3 + 6 + 3 + 3) - 3
                )  # height + num_bodies * 15 (pos + vel + rot + ang_vel) - root_pos
            else:
                raise NotImplementedError()

            if self._has_shape_obs:
                self._num_self_obs += 11
            # if self._has_limb_weight_obs: self._num_self_obs += 23 + 24 if not self._masterfoot else  29 + 30 # 23 + 24 (length + weight)
            if self._has_limb_weight_obs:
                self._num_self_obs += 10

            if not self._root_height_obs:
                self._num_self_obs -= 1

        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert False

        # if(self.ae_type == 'pvae'):
        #     #self._num_self_obs = 19 + 144 + 144 + 19 #72 + 72  # red + blue guy observation
        #     self._num_self_obs = 144 
        # else:
        self._num_self_obs = 72 + 72 + 72 + 72 + 1 

        # Account for all agents
        self._num_self_obs *= num_agents

        # TODO: To add box obs
        # self._num_self_obs += 13

        return

    def _build_termination_heights(self):
        head_term_height = 0.3
        shield_term_height = 0.32

        termination_height = self.cfg["env"]["terminationHeight"]
        self._termination_heights = np.array([termination_height] * self.num_bodies)

        head_id = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], "head"
        )
        self._termination_heights[head_id] = max(
            head_term_height, self._termination_heights[head_id]
        )

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if asset_file == "mjcf/amp_humanoid_sword_shield.xml":
            left_arm_id = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.humanoid_handles[0], "left_lower_arm"
            )
            self._termination_heights[left_arm_id] = max(
                shield_term_height, self._termination_heights[left_arm_id]
            )

        self._termination_heights = to_torch(
            self._termination_heights, device=self.device
        )
        return

    def _create_smpl_humanoid_xml(self, num_humanoids, smpl_robot, queue, pid):
        np.random.seed(np.random.randint(5002) * (pid + 1))
        res = {}
        for idx in num_humanoids:
            if self.has_shape_variation:
                gender_beta = self._amass_gender_betas[
                    idx % self._amass_gender_betas.shape[0]
                ]
            else:
                gender_beta = np.zeros(17)

            if flags.im_eval:
                gender_beta = np.zeros(17)

            asset_id = uuid4()
            asset_file_real = f"/tmp/smpl/smpl_humanoid_{asset_id}.xml"

            smpl_robot.load_from_skeleton(
                betas=torch.from_numpy(gender_beta[None, 1:]),
                gender=gender_beta[0:1],
                objs_info=None,
            )
            smpl_robot.write_xml(asset_file_real)

            res[idx] = (gender_beta, asset_file_real)

        if not queue is None:
            queue.put(res)
        else:
            return res

    def _load_amass_gender_betas(self):
        if self._has_mesh:
            gender_betas_data = joblib.load("sample_data/amass_isaac_gender_betas.pkl")
            self._amass_gender_betas = np.array(list(gender_betas_data.values()))
        else:
            gender_betas_data = joblib.load(
                "sample_data/amass_isaac_gender_betas_unique.pkl"
            )
            self._amass_gender_betas = np.array(gender_betas_data)

    def _create_envs(self, num_envs, spacing, num_per_row):

        # Add marker assets
        self._marker_handles = [[] for _ in range(num_envs)]
        self._load_marker_asset()

        # Add projectile assets
        self._proj_handles = []
        self._load_proj_asset()

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        self.humanoid_masses = []

        if self.smpl_humanoid:
            self.humanoid_shapes = []
            self.humanoid_assets = []
            self.humanoid_limb_and_weights = []
            self.skeleton_trees = []
            robot_cfg = {
                "mesh": self._has_mesh,
                "replace_feet": self._replace_feet,
                "rel_joint_lm": self._has_jt_limit,
                "upright_start": self._has_upright_start,
                "remove_toe": self.remove_toe,
                "freeze_hand": self._freeze_hand,
                "real_weight_porpotion_capsules": self._real_weight_porpotion_capsules,
                "real_weight": self._real_weight,
                "masterfoot": self._masterfoot,
                "master_range": self._master_range,
                "big_ankle": self.big_ankle,
                "box_body": self._box_body,
                "model": "smpl",
                "body_params": {},
                "joint_params": {},
                "geom_params": {},
                "actuator_params": {},
            }
            robot = SMPL_Robot(
                robot_cfg,
                data_dir="data/smpl",
            )

            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            if self.has_shape_variation:
                queue = multiprocessing.Queue()
                num_jobs = multiprocessing.cpu_count()
                if num_jobs <= 8:
                    num_jobs = 1
                if flags.debug:
                    num_jobs = 1
                res_acc = {}
                jobs = np.arange(num_envs)
                chunk = np.ceil(len(jobs) / num_jobs).astype(int)
                jobs = [jobs[i : i + chunk] for i in range(0, len(jobs), chunk)]
                job_args = [jobs[i] for i in range(len(jobs))]

                for i in range(1, len(jobs)):
                    worker_args = (job_args[i], robot, queue, i)
                    worker = multiprocessing.Process(
                        target=self._create_smpl_humanoid_xml, args=worker_args
                    )
                    worker.start()
                res_acc.update(self._create_smpl_humanoid_xml(jobs[0], robot, None, 0))
                for i in tqdm(range(len(jobs) - 1)):
                    res = queue.get()
                    res_acc.update(res)

                # if flags.debug:
                # asset_options.fix_base_link = True

                for idx in np.arange(num_envs):
                    gender_beta, asset_file_real = res_acc[idx]
                    humanoid_asset = self.gym.load_asset(
                        self.sim, asset_root, asset_file_real, asset_options
                    )
                    actuator_props = self.gym.get_asset_actuator_properties(
                        humanoid_asset
                    )
                    motor_efforts = [prop.motor_effort for prop in actuator_props]

                    self.sk_tree = SkeletonTree.from_mjcf(asset_file_real)

                    # create force sensors at the feet
                    right_foot_idx = self.gym.find_asset_rigid_body_index(
                        humanoid_asset, "L_Ankle"
                    )
                    left_foot_idx = self.gym.find_asset_rigid_body_index(
                        humanoid_asset, "R_Ankle"
                    )
                    sensor_pose = gymapi.Transform()

                    self.gym.create_asset_force_sensor(
                        humanoid_asset, right_foot_idx, sensor_pose
                    )
                    self.gym.create_asset_force_sensor(
                        humanoid_asset, left_foot_idx, sensor_pose
                    )
                    for _ in range(num_agents):
                        self.humanoid_shapes.append(
                            torch.from_numpy(gender_beta).float()
                        )
                        self.humanoid_assets.append(humanoid_asset)
                        self.skeleton_trees.append(sk_tree)

                robot.remove_geoms()  # Clean up the geoms
                self.humanoid_shapes = torch.vstack(self.humanoid_shapes).to(
                    self.device
                )
            else:
                gender_beta, asset_file_real = self._create_smpl_humanoid_xml(
                    [0], robot, None, 0
                )[0]
                asset_root = "src/phc/data/assets/mjcf/"
                asset_file_real = "smpl_humanoid_real2.xml"
                sk_tree = SkeletonTree.from_mjcf(
                    os.path.join(asset_root, asset_file_real)
                )

                humanoid_asset = self.gym.load_asset(
                    self.sim, asset_root, asset_file_real, asset_options
                )
                actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
                motor_efforts = [prop.motor_effort for prop in actuator_props]

                # create force sensors at the feet
                right_foot_idx = self.gym.find_asset_rigid_body_index(
                    humanoid_asset, "right_foot"
                )
                left_foot_idx = self.gym.find_asset_rigid_body_index(
                    humanoid_asset, "left_foot"
                )
                sensor_pose = gymapi.Transform()

                self.gym.create_asset_force_sensor(
                    humanoid_asset, right_foot_idx, sensor_pose
                )
                self.gym.create_asset_force_sensor(
                    humanoid_asset, left_foot_idx, sensor_pose
                )
                self.humanoid_shapes = (
                    torch.tensor(np.array([gender_beta] * num_envs * num_agents))
                    .float()
                    .to(self.device)
                )
                self.humanoid_assets = [humanoid_asset] * num_envs * num_agents
                self.skeleton_trees = [sk_tree] * num_envs * num_agents

        else:

            asset_path = os.path.join(asset_root, asset_file)
            asset_root = os.path.dirname(asset_path)
            asset_file = os.path.basename(asset_path)

            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            # asset_options.fix_base_link = True
            humanoid_asset = self.gym.load_asset(
                self.sim, asset_root, asset_file, asset_options
            )

            actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
            motor_efforts = [prop.motor_effort for prop in actuator_props]

            # create force sensors at the feet
            right_foot_idx = self.gym.find_asset_rigid_body_index(
                humanoid_asset, "right_foot"
            )
            left_foot_idx = self.gym.find_asset_rigid_body_index(
                humanoid_asset, "left_foot"
            )
            sensor_pose = gymapi.Transform()

            self.gym.create_asset_force_sensor(
                humanoid_asset, right_foot_idx, sensor_pose
            )
            self.gym.create_asset_force_sensor(
                humanoid_asset, left_foot_idx, sensor_pose
            )
            self.humanoid_assets = [humanoid_asset] * num_envs

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)
        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)

        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_asset_joints = self.gym.get_asset_joint_count(humanoid_asset)
        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, self.humanoid_assets[i])
            self.envs.append(env_ptr)
        self.humanoid_limb_and_weights = torch.stack(self.humanoid_limb_and_weights).to(
            self.device
        )
        self.additive_agent_pos = (
            torch.cat(agent_pos, dim=0).view(self.num_envs, num_agents, -1).clone()
        )
        self.additive_agent_pos = (
            self.additive_agent_pos.unsqueeze(2)
            .repeat(1, 1, 24, 1)
            .reshape(self.num_envs, num_agents * 24, 3)
        )
        self.initial_additive_agent_pos = self.additive_agent_pos.clone()

        print("Humanoid Weights", self.humanoid_masses[:10])

        dof_prop = self.gym.get_actor_dof_properties(
            self.envs[0], self.humanoid_handles[0]
        )

        ######################################## Joint frictino
        # dof_prop['friction'][:] = 10
        # self.gym.set_actor_dof_properties(self.envs[0], self.humanoid_handles[0], dof_prop)

        for j in range(self.num_dof):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                self.dof_limits_lower.append(dof_prop["upper"][j])
                self.dof_limits_upper.append(dof_prop["lower"][j])
            else:
                self.dof_limits_lower.append(dof_prop["lower"][j])
                self.dof_limits_upper.append(dof_prop["upper"][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if self._pd_control:
            self._build_pd_action_offset_scale()

        return

    def _load_marker_asset(self):
        asset_root = "src/phc/data/assets/mjcf/"
        asset_file = "traj_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.density = 0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        # if self._divide_group or flags.divide_group:
        #     col_group = self._group_ids[env_id]
        # else:
        #     col_group = env_id  # no inter-environment collision

        for i in range(num_agents):
            col_group = env_id  # no inter-environment collision

            col_filter = 0
            if (self.smpl_humanoid) and (not self._has_self_collision):
                col_filter = 1

            start_pose = gymapi.Transform()
            asset_file = self.cfg["env"]["asset"]["assetFileName"]
            # if (asset_file == "mjcf/ov_humanoid.xml" or asset_file == "mjcf/ov_humanoid_sword_shield.xml"):
            #     char_h = 0.927
            # else:
            #     char_h = 0.89
            char_h = 0.85
            if self.actor_init_pos == "static":
                if num_agents > 1 and (i - 1) % num_agents == 0:  # second agent
                    pos = agent_pos[
                        env_id * num_agents
                    ].clone()  # get the pos of first agent in this environment
                    pos[0] -= 3
                    pos[1] += 3  # offset the position
                else:
                    pos = torch.tensor(get_axis_params(char_h, self.up_axis_idx)).to(
                        self.device
                    )
            if self.actor_init_pos == "random":

                pos = torch.tensor(get_axis_params(char_h, self.up_axis_idx)).to(
                    self.device
                )
                pos[:2] += (
                    torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
                    * 6
                )

            elif self.actor_init_pos == "back_to_back":

                if (i - 1) % num_agents == 0:  # second agent
                    pos = agent_pos[
                        env_id * num_agents
                    ].clone()  # get the pos of first agent in this environment
                    # pos[0] -= 0.1
                    pos[1] += 0.5  # offset the position
                    q = quat_from_angle_axis(
                        torch.tensor([-1.5 * np.pi]), torch.tensor([0.0, 0.0, 1.0])
                    )
                    # start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) * gymapi.Quat(q[0][0],q[0][1],q[0][2],q[0][3])
                else:
                    pos = torch.tensor(get_axis_params(char_h, self.up_axis_idx)).to(
                        self.device
                    )
                    q = quat_from_angle_axis(
                        torch.tensor([-0.5 * np.pi]), torch.tensor([0.0, 0.0, 1.0])
                    )
                    # start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) * gymapi.Quat(q[0][0],q[0][1],q[0][2],q[0][3])

            agent_pos.append(pos)
            start_pose.p = gymapi.Vec3(*pos)

            humanoid_handle = self.gym.create_actor(
                env_ptr,
                humanoid_asset,
                start_pose,
                "humanoid{:d}".format(i),
                col_group,
                col_filter,
                0,
            )
            self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)
            mass_ind = [
                prop.mass
                for prop in self.gym.get_actor_rigid_body_properties(
                    env_ptr, humanoid_handle
                )
            ]
            humanoid_mass = np.sum(mass_ind)
            self.humanoid_masses.append(humanoid_mass)

            curr_skeleton_tree = self.skeleton_trees[env_id]
            limb_lengths = torch.norm(curr_skeleton_tree.local_translation, dim=-1)
            masses = torch.tensor(mass_ind)

            # humanoid_limb_weight = torch.cat([limb_lengths[1:], masses])

            limb_lengths = [
                limb_lengths[group].sum() for group in self.limb_weight_group
            ]
            masses = [masses[group].sum() for group in self.limb_weight_group]
            humanoid_limb_weight = torch.tensor(limb_lengths + masses)
            self.humanoid_limb_and_weights.append(
                humanoid_limb_weight
            )  # ZL: attach limb lengths and full body weight.

            percentage = 1 - np.clip((humanoid_mass - 70) / 70, 0, 1)
            if i == 0:
                color_vec = gymapi.Vec3(*get_color_gradient(percentage, "Greens"))
            elif i == 1:
                color_vec = gymapi.Vec3(*get_color_gradient(percentage, "Reds"))
            else:
                color_vec = gymapi.Vec3(*get_color_gradient(percentage, "Blues"))

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, color_vec
                )

            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            if self._pd_control:
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                if self.has_shape_variation:
                    pd_scale = humanoid_mass / self.cfg["env"].get(
                        "default_humanoid_mass", 77.0 if self._real_weight else 35.0
                    )
                    dof_prop["stiffness"] *= pd_scale * self._kp_scale
                    dof_prop["damping"] *= pd_scale * self._kd_scale

            else:
                dof_prop["driveMode"] = gymapi.DOF_MODE_EFFORT
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

            if self.smpl_humanoid and self._has_self_collision:
                # compliance_vals = [0.1] * 24
                # thickness_vals = [1.0] * 24
                if self._has_mesh:
                    if self._masterfoot:
                        filter_ints = [
                            0,
                            1,
                            96,
                            192,
                            192,
                            192,
                            192,
                            192,
                            192,
                            192,
                            1,
                            384,
                            288,
                            288,
                            288,
                            288,
                            288,
                            288,
                            288,
                            1024,
                            6,
                            1560,
                            0,
                            512,
                            0,
                            20,
                            0,
                            0,
                            0,
                            0,
                            10,
                            0,
                            0,
                            0,
                        ]
                    else:
                        filter_ints = [
                            0,
                            1,
                            224,
                            512,
                            384,
                            1,
                            1792,
                            64,
                            1056,
                            4096,
                            6,
                            6168,
                            0,
                            2048,
                            0,
                            20,
                            0,
                            0,
                            0,
                            0,
                            10,
                            0,
                            0,
                            8192,
                        ]
                else:
                    if self._masterfoot:
                        # filter_ints = [0, 0, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 12, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 32, 0, 48, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        filter_ints = [
                            0,
                            0,
                            3,
                            6,
                            6,
                            6,
                            6,
                            6,
                            6,
                            6,
                            0,
                            12,
                            9,
                            9,
                            9,
                            9,
                            9,
                            9,
                            9,
                            32,
                            0,
                            48,
                            0,
                            16,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    else:
                        filter_ints = np.array(
                            [
                                0,
                                0,
                                7,
                                16,
                                12,
                                0,
                                56,
                                2,
                                33,
                                128,
                                0,
                                192,
                                0,
                                64,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                            ]
                        )
                        # zero_yes = filter_ints == 0
                        # to_put = 2 ** np.arange(np.sum(zero_yes))
                        # filter_ints[zero_yes] = to_put
                        # filter_ints[~zero_yes] *= to_put.max()
                        filter_ints[:] = 0
                        filter_ints = filter_ints.tolist()

                        # filter_ints = np.array([1, 2, 7, 16, 12, 3, 56, 2, 33, 128, 4, 192, 5, 64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] + 32)
                        # filter_ints = [32, 33, 7, 16, 12, 34, 56, 2, 33, 128, 35, 192, 36, 64, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
                        # filter_ints = [32, 33, 7, 16, 12, 3, 56, 2, 33, 128, 4, 192, 5, 64, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                        # filter_ints = [1, 2, 7 * 32768, 16 * 32768, 12 * 32768, 4, 56 * 32768, 2 * 32768, 33 * 32768, 128 * 32768, 8, 192 * 32768, 16, 64 * 32768, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 0]
                        # filter_ints = [1, 1, 14, 32, 24, 1, 112, 4, 66, 256, 1, 384, 0, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                props = self.gym.get_actor_rigid_shape_properties(
                    env_ptr, humanoid_handle
                )

                assert len(filter_ints) == len(props)
                for p_idx in range(len(props)):
                    props[p_idx].filter = filter_ints[p_idx]
                self.gym.set_actor_rigid_shape_properties(
                    env_ptr, humanoid_handle, props
                )

            self.humanoid_handles.append(humanoid_handle)

        for i in range(num_agents):
            # Add marker
            color = gymapi.Vec3(0.8, 0.0, 0.0)
            self._build_marker(env_id, env_ptr, color)

        for i in range(num_agents):
            # Add marker
            color = gymapi.Vec3(0.0, 0.0, 1.0)
            self._build_marker(env_id, env_ptr, color)

        # Add external object
        self._build_proj(env_id, env_ptr, 4)

        return

    def _build_marker(self, env_id, env_ptr, color):
        default_pose = gymapi.Transform()
        pos = [
            [-0.01, 0.0, 0.0],
            # [ 0.0890016, -0.40830246, 0.25]
        ]
        # pos[:2] += pos_add[:2]
        default_pose = gymapi.Transform()
        torch.manual_seed(int(time.time()))
        # default_pose.p.x = 5
        # default_pose.p.y = pos[0][1]

        for i in range(self._num_joints):
            marker_handle = self.gym.create_actor(
                env_ptr,
                self._marker_asset,
                default_pose,
                "marker",
                self.num_envs + 10,
                1,
                0,
            )
            if i in self._track_bodies_id:
                self.gym.set_rigid_body_color(
                    env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, color
                )
            else:
                self.gym.set_rigid_body_color(
                    env_ptr,
                    marker_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(1.0, 1.0, 1.0),
                )
            self._marker_handles[env_id].append(marker_handle)

        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        # these variables are pointers to where marker root state is located
        self._marker_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., num_agents : (num_agents + num_agents * self._num_joints * 2), :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_rotation = self._marker_states[..., 3:7]

        self._marker_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(
            self._marker_handles, dtype=torch.int32, device=self.device
        )
        self._marker_actor_ids = self._marker_actor_ids.flatten()

        return

    def _build_box_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._box_states = self._root_states_reshaped[..., -1, :]
        self._box_pos = self._box_states[..., :3]
        self._box_rotation = self._box_states[..., 3:7]

        self._box_actor_ids = self._humanoid_actor_ids + to_torch(
            self._proj_handles, dtype=torch.int32, device=self.device
        )
        self._box_actor_ids = self._box_actor_ids.flatten()

    def _build_pd_action_offset_scale(self):
        num_joints = len(self._dof_offsets) - 1

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]
            if not self._bias_offset:
                if dof_size == 3:
                    curr_low = lim_low[dof_offset : (dof_offset + dof_size)]
                    curr_high = lim_high[dof_offset : (dof_offset + dof_size)]
                    curr_low = np.max(np.abs(curr_low))
                    curr_high = np.max(np.abs(curr_high))
                    curr_scale = max([curr_low, curr_high])
                    curr_scale = 1.2 * curr_scale
                    curr_scale = min([curr_scale, np.pi])

                    lim_low[dof_offset : (dof_offset + dof_size)] = -curr_scale
                    lim_high[dof_offset : (dof_offset + dof_size)] = curr_scale

                    # lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                    # lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

                elif dof_size == 1:
                    curr_low = lim_low[dof_offset]
                    curr_high = lim_high[dof_offset]
                    curr_mid = 0.5 * (curr_high + curr_low)

                    # extend the action range to be a bit beyond the joint limits so that the motors
                    # don't lose their strength as they approach the joint limits
                    curr_scale = 0.7 * (curr_high - curr_low)
                    curr_low = curr_mid - curr_scale
                    curr_high = curr_mid + curr_scale

                    lim_low[dof_offset] = curr_low
                    lim_high[dof_offset] = curr_high
            else:
                curr_low = lim_low[dof_offset : (dof_offset + dof_size)]
                curr_high = lim_high[dof_offset : (dof_offset + dof_size)]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset : (dof_offset + dof_size)] = curr_low
                lim_high[dof_offset : (dof_offset + dof_size)] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)
        if self.smpl_humanoid:
            self._L_knee_dof_idx = self._dof_names.index("L_Knee") * 3 + 1
            self._R_knee_dof_idx = self._dof_names.index("R_Knee") * 3 + 1

            # ZL: Modified SMPL to give stronger knee
            self._pd_action_scale[self._L_knee_dof_idx] = 5
            self._pd_action_scale[self._R_knee_dof_idx] = 5

            if self._has_smpl_pd_offset:
                if self._has_upright_start:
                    self._pd_action_offset[self._dof_names.index("L_Shoulder") * 3] = (
                        -np.pi / 2
                    )
                    self._pd_action_offset[self._dof_names.index("R_Shoulder") * 3] = (
                        np.pi / 2
                    )
                else:
                    self._pd_action_offset[self._dof_names.index("L_Shoulder") * 3] = (
                        -np.pi / 6
                    )
                    self._pd_action_offset[
                        self._dof_names.index("L_Shoulder") * 3 + 2
                    ] = (-np.pi / 2)
                    self._pd_action_offset[self._dof_names.index("R_Shoulder") * 3] = (
                        -np.pi / 3
                    )
                    self._pd_action_offset[
                        self._dof_names.index("R_Shoulder") * 3 + 2
                    ] = (np.pi / 2)

        return

    def _compute_reward(self, actions):
        # dof_names = [ 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'L_Toe_1', 'L_Toe_1_1', 'L_Toe_2', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'R_Toe_1', 'R_Toe_1_1', 'R_Toe_2', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
        #        'R_Elbow', 'R_Wrist', 'R_Hand']
        hand_idx = self._body_names.index("R_Hand")
        head_idx = self._body_names.index("Head")
        if(self.first_cam_update):
            self.first_cam_update = False
            self._init_camera()
            self._update_camera()

        # TODO: THese all should be self._rigid_body_pos !?
        # first_char_hands_pos = self.ref_body_pos[:,[self._body_names.index("R_Hand"),self._body_names.index("L_Hand")],:].clone()

        # second_char_full_body_pos = self.ref_body_pos[:,24:,:].clone()
        # print(self.step_count)

        # NOTE: the compute reward is called after calculate observation, so the self.blue_obs and self.red_obs are valid
        chain_list_indecies = []
        chain_list = ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]
        for i in chain_list:
            chain_list_indecies.append(self._body_names.index(i))

        for i in range(self.input_lats.shape[-1]):

            self.plot_scaler(
                    f'action/action_{i}',
                    torch.mean(self.input_lats[...,i]),
                    self.step_count,
                )

        self.rew_buf[:] = compute_humanoid_reward(
            self,
            self._rigid_body_pos.reshape(self.num_envs, -1),
            self.red_rb_xyz.reshape(self.num_envs, -1),
            self.blue_rb_xyz.reshape(self.num_envs, -1),
            self.red_rb_root_xyz.reshape(self.num_envs, -1),
            self.prev_red_rb_root_xyz.reshape(self.num_envs, -1),
            self.blue_rb_root_xyz.reshape(self.num_envs, -1),
            self.prev_blue_rb_root_xyz.reshape(self.num_envs, -1),
            self._box_pos.reshape(self.num_envs, -1),
            hand_idx,
            head_idx)
        self.rew_buf[:] = self.rew_buf[:] * (1.0 - self._terminate_buf)
        return

    def _compute_reset(self):
        hand_idx = self._body_names.index("R_Hand")
        head_idx = self._body_names.index("Head")
        
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.step_count,
            hand_idx,
            head_idx,
            self.reset_buf,
            self.progress_buf,
            self._contact_forces,
            self._contact_body_ids,
            self._rigid_body_pos.reshape(self.num_envs, num_agents * self.num_bodies, 3),
            self._termination_heights, 
            self.blue_rb_xyz,
            self.red_rb_xyz,
            self.ref_body_root_rot,
            self.red_rb_root_xyz,
            self.prev_red_rb_root_xyz,
            self._box_pos,
            self.motion_times,
            self._motion_lib._motion_lengths.unsqueeze(-1),
            self.max_episode_length,
            self._enable_early_termination,
        )
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        return

    def _compute_observations(
        self, r_body_pos, r_body_rot, r_body_vel, r_body_ang_vel, env_ids=None
    ):

        # TODO: fix the function to use env_ids
        if(self.physics_enable):
            obs = self._compute_humanoid_obs(
                r_body_pos, r_body_rot, r_body_vel, r_body_ang_vel, env_ids
            )

        # TODO: NOTE: For now i'm having issue with concating multi character obs with box, so i'm just ignoring box observation

        # Concatenate box state
        # obs = torch.cat([obs, self._box_states[env_ids]], dim=-1)

        # self.blue_obs =  torch.cat([self.blue_rb_root_xyz, self.blue_rb_root_rot_sixd, self.blue_rb_root_vel
        #         ,self.blue_rb_root_ang_vel, self.blue_sixd, self.blue_joint_ang_vel], dim=-1)
        # self.red_obs =  torch.cat([self.red_rb_root_xyz, self.red_rb_root_rot_sixd, self.red_rb_root_vel,
        #                         self.red_rb_root_ang_vel, self.red_sixd, self.red_joint_ang_vel], dim=-1)
        #else:
        self.blue_obs = torch.cat([ self.blue_rb_xyz, self.prev_blue_rb_xyz], dim=-1)
        #self.red_obs = torch.cat([self.red_rb_xyz,],dim=-1,)
        # self.red_obs = torch.cat([self._rigid_body_pos.reshape(self.num_envs, -1), 
        #                           self.red_rb_xyz.reshape(self.num_envs, -1), 
        #                           self._box_pos.reshape(self.num_envs, -1), 
        #                         (self.red_rb_xyz.reshape(-1,24,3) - self._box_pos[...,None,:]).reshape(self.num_envs, -1),],dim=-1,)
        self.red_obs = torch.cat([self.current_red_yaw_angles, self.red_rb_xyz,self.prev_red_rb_xyz],dim=-1,)
        #print(self.red_obs.shape)

        if env_ids is None:
            if(self.physics_enable):
                new_obs = obs.clone().reshape(self.num_envs, -1)
            # self.obs_buf[:] = torch.cat([new_obs[...,358:]], dim=-1)

            self.obs_buf[:] = torch.cat([self.blue_obs, self.red_obs], dim=-1).reshape(
                self.obs_buf.shape[0], -1
            )
            # self.obs_buf = self.obs_buf
        else:
            if(self.physics_enable):
                new_obs = obs.clone().reshape(self.num_envs, -1)
            # self.obs_buf[env_ids] = torch.cat([new_obs[...,358:]], dim=-1)
            # TODO fix this to use env ids
            self.obs_buf[:] = torch.cat([self.blue_obs, self.red_obs], dim=-1).reshape(
                self.obs_buf.shape[0], -1
            )
        # print(self.obs_buf)
        # NOTE this should be obs if we use it with phc
        if(self.physics_enable):
            return new_obs
        else:
            return self.obs_buf

    # TODO: fix the function to use env_ids
    def _compute_humanoid_obs(
        self, r_body_pos, r_body_rot, r_body_vel, r_body_ang_vel, env_ids=None
    ):
        if ENABLE_MAX_COORD_OBS:
            if env_ids is None:
                body_pos = r_body_pos
                body_rot = r_body_rot
                body_vel = r_body_vel
                body_ang_vel = r_body_ang_vel

                if self.self_obs_v == 2:
                    body_pos = torch.cat(
                        [self._rigid_body_pos_hist, body_pos.unsqueeze(1)], dim=1
                    )
                    body_rot = torch.cat(
                        [self._rigid_body_rot_hist, body_rot.unsqueeze(1)], dim=1
                    )
                    body_vel = torch.cat(
                        [self._rigid_body_vel_hist, body_vel.unsqueeze(1)], dim=1
                    )
                    body_ang_vel = torch.cat(
                        [self._rigid_body_ang_vel_hist, body_ang_vel.unsqueeze(1)],
                        dim=1,
                    )

            else:
                # TODO: to fix for env_ids, prob should reshape here and then reshape after
                body_pos = r_body_pos[env_ids]
                body_rot = r_body_rot[env_ids]
                body_vel = r_body_vel[env_ids]
                body_ang_vel = r_body_ang_vel[env_ids]

                if self.self_obs_v == 2:
                    body_pos = torch.cat(
                        [self._rigid_body_pos_hist[env_ids], body_pos.unsqueeze(1)],
                        dim=1,
                    )
                    body_rot = torch.cat(
                        [self._rigid_body_rot_hist[env_ids], body_rot.unsqueeze(1)],
                        dim=1,
                    )
                    body_vel = torch.cat(
                        [self._rigid_body_vel_hist[env_ids], body_vel.unsqueeze(1)],
                        dim=1,
                    )
                    body_ang_vel = torch.cat(
                        [
                            self._rigid_body_ang_vel_hist[env_ids],
                            body_ang_vel.unsqueeze(1),
                        ],
                        dim=1,
                    )

            if self.smpl_humanoid:
                if env_ids is None:
                    body_shape_params = (
                        self.humanoid_shapes[:, :-6]
                        if self.smpl_humanoid
                        else self.humanoid_shapes
                    )
                    limb_weights = self.humanoid_limb_and_weights
                else:
                    body_shape_params = (
                        self.humanoid_shapes[env_ids, :-6]
                        if self.smpl_humanoid
                        else self.humanoid_shapes[env_ids]
                    )
                    limb_weights = self.humanoid_limb_and_weights[env_ids]

                if self.self_obs_v == 1:
                    obs = compute_humanoid_observations_smpl_max(
                        body_pos,
                        body_rot,
                        body_vel,
                        body_ang_vel,
                        body_shape_params,
                        limb_weights,
                        self._local_root_obs,
                        self._root_height_obs,
                        self._has_upright_start,
                        self._has_shape_obs,
                        self._has_limb_weight_obs,
                    )
                elif self.self_obs_v == 2:
                    obs = compute_humanoid_observations_smpl_max_v2(
                        body_pos,
                        body_rot,
                        body_vel,
                        body_ang_vel,
                        body_shape_params,
                        limb_weights,
                        self._local_root_obs,
                        self._root_height_obs,
                        self._has_upright_start,
                        self._has_shape_obs,
                        self._has_limb_weight_obs,
                        self.past_track_steps + 1,
                    )

            else:
                obs = compute_humanoid_observations_max(
                    body_pos,
                    body_rot,
                    body_vel,
                    body_ang_vel,
                    self._local_root_obs,
                    self._root_height_obs,
                )

        else:
            if env_ids is None:
                root_pos = r_body_pos[:, 0, :]
                root_rot = r_body_rot[:, 0, :]
                root_vel = r_body_vel[:, 0, :]
                root_ang_vel = r_body_ang_vel[:, 0, :]
                dof_pos = self._dof_pos[..., agent]
                dof_vel = self._dof_vel[..., agent]
                key_body_pos = r_body_pos[:, self._key_body_ids, :]
            else:
                root_pos = r_body_pos[env_ids][:, 0, :]
                root_rot = r_body_rot[env_ids][:, 0, :]
                root_vel = r_body_vel[env_ids][:, 0, :]
                root_ang_vel = r_body_ang_vel[env_ids][:, 0, :]
                dof_pos = self._dof_pos[env_ids, agent]
                dof_vel = self._dof_vel[env_ids, agent]
                key_body_pos = r_body_pos[env_ids][:, self._key_body_ids, :]

            if (self.smpl_humanoid) and self.self.has_shape_obs:
                if env_ids is None:
                    body_shape_params = self.humanoid_shapes
                else:
                    body_shape_params = self.humanoid_shapes[env_ids]
                obs = compute_humanoid_observations_smpl(
                    root_pos,
                    root_rot,
                    root_vel,
                    root_ang_vel,
                    dof_pos,
                    dof_vel,
                    key_body_pos,
                    self._dof_obs_size,
                    self._dof_offsets,
                    body_shape_params,
                    self._local_root_obs,
                    self._root_height_obs,
                    self._has_upright_start,
                    self._has_shape_obs,
                )
            else:
                obs = compute_humanoid_observations(
                    root_pos,
                    root_rot,
                    root_vel,
                    root_ang_vel,
                    dof_pos,
                    dof_vel,
                    key_body_pos,
                    self._local_root_obs,
                    self._root_height_obs,
                    self._dof_obs_size,
                    self._dof_offsets,
                )
        return obs

    def _reset_actors(self, env_ids):
        # d = torch.load("fallen.pkl")
        #
        # self._humanoid_root_states[env_ids] = d["root_states"]
        # self._dof_pos[env_ids] = d["dof_pos"]
        # self._dof_vel[env_ids] = d["dof_vel"]
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[
            env_ids
        ].clone()
        self._reset_ref_state_init(env_ids)
        self._corrected_initial_humanoid_root_pos = self._humanoid_root_states.clone()[
            ..., 0:2
        ]
        # print(self._corrected_initial_humanoid_root_pos)
        self.obs_root_pos = self._humanoid_root_states.clone()[..., 0:3]
        # Original

        # self._ += self.additive_agent_pos[...,:2]
        self.additive_agent_pos = self.initial_additive_agent_pos.clone()
        # self._dof_pos[env_ids] = self._initial_dof_pos[env_ids].clone()
        # self._dof_vel[env_ids] = self._initial_dof_vel[env_ids].clone()

        self._box_states[env_ids] = self._initial_box_states[env_ids].clone()

        # self._set_env_state(
        #     env_ids=env_ids,
        #     root_pos=self._initial_humanoid_root_states[:, :3],
        #     root_rot=self._initial_humanoid_root_states[:, 3:7],
        #     dof_pos=self._initial_dof_pos,
        #     root_vel=self._initial_humanoid_root_states[:, 7:10],
        #     root_ang_vel=self._initial_humanoid_root_states[:, 10:13],
        #     dof_vel=self._initial_dof_vel,
        #     rigid_body_pos=self._initial_rigid_body_state[:, ],
        #     rigid_body_rot=rb_rot,
        #     rigid_body_vel=body_vel,
        #     rigid_body_ang_vel=body_ang_vel,
        # )
        return

    def _cache_anim_root(self, env_ids):
        # self._load_motion(self.motion_file)
        # self._motion_start_times_offset[env_ids] *= 0  # Reset the motion time offsets
        # self._global_offset[env_ids] *= 0  # Reset the global offset when resampling.
        # self._cycle_counter[env_ids] = 0

        num_envs = env_ids.shape[0]
        (
            motion_ids,
            motion_times,
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            rb_pos,
            rb_rot,
            body_vel,
            body_ang_vel,
        ) = self._sample_ref_state(env_ids)
        self.modified_rb_body_pos = rb_pos.clone()
        self.ref_body_vel = body_vel.clone()
        self.ref_rb_rot = rb_rot.clone()
        self.ref_body_ang_vel = body_ang_vel.clone()

        self.anim_root_rot_yaw_cache = torch_utils.get_euler_xyz(root_rot)[2]
        start_indices = np.arange(self.num_envs) * num_agents
        end_indices = np.arange(self.num_envs) * num_agents + 1
        # Slice the cache array and perform the calculation
        if self.actor_init_pos == "back_to_back":
            self.new_angles = (
                self.anim_root_rot_yaw_cache[start_indices]
                + np.pi
                - self.anim_root_rot_yaw_cache[end_indices]
            )
            self.new_angles = self.new_angles.unsqueeze(1).repeat(1, 24).unsqueeze(-1)

    def _reset_ref_state_init(self, env_ids):
        # self._load_motion(self.motion_file)
        self._motion_start_times_offset[env_ids] *= 0  # Reset the motion time offsets
        self._global_offset[env_ids] *= 0  # Reset the global offset when resampling.
        # self._cycle_counter[env_ids] = 0
        # print(len(env_ids))

        num_envs = env_ids.shape[0]
        (
            motion_ids,
            motion_times,
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            rb_pos,
            rb_rot,
            body_vel,
            body_ang_vel,
        ) = self._sample_ref_state(env_ids)

        rb_rot_sixd = torch.tensor(
            Rotation.from_quat(rb_rot.reshape(-1, 4))
            .as_matrix()
            .reshape((*rb_rot.shape[:-1], 3, 3))[..., :2]
            .reshape((*rb_rot.shape[:-1], 6)),
            device=self.device,
            dtype=torch.float,
        )
        if(self.ae_type == 'pvae'):

            rb_rot_sixd_inv, root_yaw = remove_root_yaw_from_sixd(rb_rot_sixd)
            xs = rb_rot_sixd_inv
            # xs = torch.tensor(xs, dtype=torch.float, device=self.device)
            xs = xs.reshape(xs.shape[0],24, -1)
            _, decoded, mu, log_var = self.ae.forward(
                xs,
                train_yes=False,
            )

        root_yaw =  torch.tensor(Rotation.from_quat(root_rot).as_euler('zyx')[...,[0]], dtype=torch.float, device=self.device)


        # TODO: is this correct? setting both observation as the ref anim?
        if len(env_ids) == self.num_envs:
            if(self.ae_type == 'pvae'):
                self.blue_latent = mu * 1
                self.blue_sixd_inv = rb_rot_sixd_inv.reshape(-1,144) * 1

                self.red_latent = self.blue_latent * 1
                self.red_sixd_inv = self.blue_sixd_inv * 1
            
            self.blue_dof_pos = dof_pos.reshape(self.num_envs * num_agents, -1) * 1
            self.red_dof_pos = self.blue_dof_pos * 1

            self.blue_rb_xyz = rb_pos.reshape(self.num_envs * num_agents, -1) * 1
            self.prev_blue_rb_xyz = self.blue_rb_xyz
            self.blue_rb_root_xyz = rb_pos[:, 0] * 1
            self.prev_blue_rb_root_xyz = self.blue_rb_root_xyz * 1
            self.blue_rb_root_rot_sixd = rb_rot_sixd[:, 0] * 1
            self.blue_rb_root_vel = body_vel[:, 0] * 1
            self.blue_rb_root_ang_vel = body_ang_vel[:, 0] * 1
            self.blue_sixd = rb_rot_sixd.reshape(self.num_envs * num_agents, -1) * 1
            self.blue_joint_ang_vel = body_ang_vel.reshape(self.num_envs * num_agents, -1) * 1
            self.ref_body_root_pos = root_pos * 1
            self.ref_body_root_rot = rb_rot_sixd[:, 0] * 1

            if(self.calc_root_dir):
                self.prev_red_yaw_angles = root_yaw *1
                self.current_red_yaw_angles = root_yaw * 1
            self.red_rb_xyz = self.blue_rb_xyz * 1
            self.prev_red_rb_xyz = self.red_rb_xyz * 1
            self.prev_red_rb_root_xyz = self.blue_rb_root_xyz * 1
            self.red_rb_root_xyz = self.blue_rb_root_xyz * 1
            self.red_rb_root_rot_sixd = self.blue_rb_root_rot_sixd * 1
            self.red_rb_root_vel = self.blue_rb_root_vel * 1
            self.red_rb_root_ang_vel = self.blue_rb_root_ang_vel * 1
            self.red_sixd = self.blue_sixd * 1
            self.red_joint_ang_vel = self.blue_joint_ang_vel * 1
        else:
            if(self.ae_type == 'pvae'):
                self.blue_latent[env_ids] = mu.reshape(env_ids.shape[0] * num_agents, -1) * 1
                self.blue_sixd_inv[env_ids] = rb_rot_sixd_inv.reshape(env_ids.shape[0] * num_agents, -1) * 1

            self.blue_dof_pos[env_ids] = dof_pos.reshape(env_ids.shape[0] * num_agents, -1) * 1
            self.red_dof_pos[env_ids] = self.blue_dof_pos[env_ids] * 1

            if(self.calc_root_dir):
                self.prev_red_yaw_angles[env_ids] = root_yaw *1
                self.current_red_yaw_angles[env_ids] = root_yaw * 1

            self.blue_rb_xyz[env_ids] = rb_pos.reshape(env_ids.shape[0] * num_agents, -1) * 1
            self.prev_blue_rb_xyz[env_ids] = self.blue_rb_xyz[env_ids]
            self.blue_rb_root_xyz[env_ids] = rb_pos[:, 0] * 1
            self.prev_blue_rb_root_xyz[env_ids] = self.blue_rb_root_xyz[env_ids]
            self.blue_rb_root_rot_sixd[env_ids] = rb_rot_sixd[:, 0] * 1
            self.blue_rb_root_vel[env_ids] = body_vel[:, 0] * 1
            self.blue_rb_root_ang_vel[env_ids] = body_ang_vel[:, 0] * 1
            self.blue_sixd[env_ids] = rb_rot_sixd.reshape(env_ids.shape[0] * num_agents, -1) * 1
            self.blue_joint_ang_vel[env_ids] = body_ang_vel.reshape(env_ids.shape[0] * num_agents, -1) * 1
            self.ref_body_root_pos[env_ids] = root_pos * 1
            self.ref_body_root_rot[env_ids] = rb_rot_sixd[:, 0] * 1

            if(self.ae_type == 'pvae'):
                self.red_latent[env_ids] = self.blue_latent[env_ids]* 1
                self.red_sixd_inv[env_ids] = self.blue_sixd_inv[env_ids]* 1
            self.red_rb_xyz[env_ids] = self.blue_rb_xyz[env_ids]* 1
            self.prev_red_rb_xyz[env_ids] = self.blue_rb_xyz[env_ids]* 1
            self.prev_red_rb_root_xyz[env_ids] = self.blue_rb_root_xyz[env_ids]* 1
            self.red_rb_root_xyz[env_ids] = self.blue_rb_root_xyz[env_ids]* 1
            self.red_rb_root_rot_sixd[env_ids] = self.blue_rb_root_rot_sixd[env_ids]* 1
            self.red_rb_root_vel[env_ids] = self.blue_rb_root_vel[env_ids]* 1
            self.red_rb_root_ang_vel[env_ids] = self.blue_rb_root_ang_vel[env_ids] * 1
            self.red_sixd[env_ids] = self.blue_sixd[env_ids] * 1
            self.red_joint_ang_vel[env_ids] = self.blue_joint_ang_vel[env_ids] * 1

        # if flags.debug:
        # print('raising for debug')
        # root_pos[..., 2] += 0.5

        # if flags.fixed:
        #     x_grid, y_grid = torch.meshgrid(torch.arange(64), torch.arange(64))
        #     root_pos[:, 0], root_pos[:, 1] = x_grid.flatten()[env_ids] * 2, y_grid.flatten()[env_ids] * 2
        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            rigid_body_pos=rb_pos,
            rigid_body_rot=rb_rot,
            rigid_body_vel=body_vel,
            rigid_body_ang_vel=body_ang_vel,
        )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        self._motion_start_times[env_ids] = motion_times.reshape(
            len(env_ids), num_agents
        )
        self._sampled_motion_ids[env_ids] = motion_ids.reshape(len(env_ids), num_agents)
        return

    def _sample_ref_state(self, env_ids):

        num_envs = env_ids.shape[0]

        """if (self._state_init == HumanoidAMP.StateInit.Random or self._state_init == HumanoidAMP.StateInit.Hybrid):
            motion_times = self._sample_time(motion_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        """
        # motion_times = torch.rand(num_envs * num_agents, device=self.device)
        motion_times = self._motion_lib.sample_time(
            self._sampled_motion_ids[env_ids].flatten().numpy()
        )
        # set start anim time to zero for evaluation
        if self.num_envs <= 5:
            motion_times *= 0

        if self.smpl_humanoid:
            motion_res = self._get_state_from_motionlib_cache(
                self._sampled_motion_ids[env_ids].flatten(),
                motion_times,
                self._global_offset[env_ids].reshape(num_envs * num_agents, 3),
            )  # TODO: change this to len(env_ids) instead of num_envs
            (
                root_pos,
                root_rot,
                dof_pos,
                root_vel,
                root_ang_vel,
                dof_vel,
                smpl_params,
                limb_weights,
                pose_aa,
                ref_rb_pos,
                ref_rb_rot,
                ref_body_vel,
                ref_body_ang_vel,
            ) = (
                motion_res["root_pos"],
                motion_res["root_rot"],
                motion_res["dof_pos"],
                motion_res["root_vel"],
                motion_res["root_ang_vel"],
                motion_res["dof_vel"],
                motion_res["motion_bodies"],
                motion_res["motion_limb_weights"],
                motion_res["motion_aa"],
                motion_res["rg_pos"],
                motion_res["rb_rot"],
                motion_res["body_vel"],
                motion_res["body_ang_vel"],
            )

        else:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
                self._motion_lib.get_motion_state(
                    self._sampled_motion_ids[env_ids].flatten(), motion_times
                )
            )
            rb_pos, rb_rot = None, None
        return (
            self._sampled_motion_ids[env_ids],
            motion_times,
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            ref_rb_pos,
            ref_rb_rot,
            ref_body_vel,
            ref_body_ang_vel,
        )

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape)

        motion_len = self._motion_lib._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len

        return motion_time

    def _set_env_state(
        self,
        env_ids,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        rigid_body_pos=None,
        rigid_body_rot=None,
        rigid_body_vel=None,
        rigid_body_ang_vel=None,
    ):

        #if self.root_motion:
        self._humanoid_root_states[env_ids, ..., 0:2] += root_pos.reshape(
            len(env_ids), num_agents, -1
        )[..., 0:2]
        self._humanoid_root_states[env_ids, ..., 3:7] = root_rot.reshape(
            len(env_ids), num_agents, -1
        )

        if self.actor_init_pos == "back_to_back":
            q = quat_from_angle_axis(
                torch.tensor([np.pi]).to(self.device),
                torch.tensor([0.0, 0.0, 1.0]).to(self.device),
            ).repeat(len(env_ids), 1)
            self._humanoid_root_states[env_ids, 1, 3:7] = quat_mul(
                q, self._humanoid_root_states[env_ids, 1, 3:7]
            )

        self._humanoid_root_states[env_ids, ..., 7:10] = root_vel.reshape(
            len(env_ids), num_agents, -1
        )
        self._humanoid_root_states[env_ids, ..., 10:13] = root_ang_vel.reshape(
            len(env_ids), num_agents, -1
        )
        self._dof_pos[env_ids] = dof_pos.reshape(len(env_ids), num_agents, -1)
        self._dof_vel[env_ids] = dof_vel.reshape(len(env_ids), num_agents, -1)

        if (not rigid_body_pos is None) and (not rigid_body_rot is None):
            self._rigid_body_pos[env_ids] = rigid_body_pos.reshape(
                len(env_ids), num_agents, 24, -1
            )
            self._rigid_body_rot[env_ids] = rigid_body_rot.reshape(
                len(env_ids), num_agents, 24, -1
            )
            self._rigid_body_vel[env_ids] = rigid_body_vel.reshape(
                len(env_ids), num_agents, 24, -1
            )
            self._rigid_body_ang_vel[env_ids] = rigid_body_ang_vel.reshape(
                len(env_ids), num_agents, 24, -1
            )

            self._reset_rb_pos = self._rigid_body_pos[env_ids].clone()
            self._reset_rb_rot = self._rigid_body_rot[env_ids].clone()
            self._reset_rb_vel = self._rigid_body_vel[env_ids].clone()
            self._reset_rb_ang_vel = self._rigid_body_ang_vel[env_ids].clone()

        return

    def pre_physics_step(self, actions):
        # if flags.debug:
        # actions *= 0
        # print(actions)
        self.motion_times = (
            (self.progress_buf.unsqueeze(1).repeat(1, num_agents)) * self.dt
            + self._motion_start_times
            + self._motion_start_times_offset
        )  # + 1 for target.
        # motion_times *=0
        # motion_times[:,1] = 0 #making sure red character stays in T pose (plays only the first frame)
        motion_res = self._get_state_from_motionlib_cache(
            self._sampled_motion_ids.flatten(),
            self.motion_times.flatten(),
            self._global_offset.reshape(self.num_envs * num_agents, 3),
        )
        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            smpl_params,
            limb_weights,
            pose_aa,
            self.ref_rb_pos,
            self.ref_rb_rot,
            self.ref_body_vel,
            self.ref_body_ang_vel,
        ) = (
            motion_res["root_pos"],
            motion_res["root_rot"],
            motion_res["dof_pos"],
            motion_res["root_vel"],
            motion_res["root_ang_vel"],
            motion_res["dof_vel"],
            motion_res["motion_bodies"],
            motion_res["motion_limb_weights"],
            motion_res["motion_aa"],
            motion_res["rg_pos"],
            motion_res["rb_rot"],
            motion_res["body_vel"],
            motion_res["body_ang_vel"],
        )

        # self.ref_body_vel *= 0
        # print(self.ref_rb_pos[:,[0],:2])
        # motion_res["rg_pos"][:, :, :2]  -=  motion_res["rg_pos"][:,[0],:2]
        # old_ref_rb_pos = self.ref_rb_pos.clone()
        # cached_root_pos =  self.ref_rb_pos[:,[0],:2].clone()
        # self.ref_rb_pos[:, :, :2]  -=  self.ref_rb_pos[:,[0],:2]

        self.full_action = (
            actions.to(self.device).clone().view(self.num_envs * num_agents, -1)
        )
        self.input_lats = self.full_action.clone()
        # print(self.input_lats)
        if len(self.input_lats.shape) == 1:
            self.input_lats = self.input_lats[None]

        if self.ae_type == "ae":
            self.my_lats = self.ae.encoder.forward(
                self.ae.rms.normalize(
                    self.ref_rb_pos.reshape(self.ref_rb_pos.shape[0], -1)
                )
            )
            self.pre_physics_step_ae(
                input_lats_importance=0,
                input_my_lats_importance=1e0,
                force_t_pose=False,
            )
        elif self.ae_type == "vae":
            self.my_lats = self.ae.encoder.forward(
                self.ae.rms.normalize(
                    self.ref_rb_pos.reshape(self.ref_rb_pos.shape[0], -1)
                )
            )
            self.pre_physics_step_vae(
                input_lats_importance=0,
                input_my_lats_importance=1e0,
                force_t_pose=False,
            )
        elif self.ae_type == "cvae":

            self.pre_physics_step_cvae(
                motion_res,
                input_lats_importance=1e0,
                input_my_lats_importance=1e0,
                force_t_pose=False,
            )
        elif self.ae_type == "pvae":
            self.pre_physics_step_pvae(
                motion_res,
                input_lats_importance=1e0,
                input_my_lats_importance=1e0,
            )
        elif self.ae_type == "pvae_dof":
            self.pre_physics_step_pvae_dof(
                motion_res,
                input_lats_importance=1e0,
                input_my_lats_importance=1e0,
            )
        elif self.ae_type == "dof":
            self.pre_physics_step_dof(
                motion_res,
                input_lats_importance=1e0,
                input_my_lats_importance=1e0,
            )
        else:
            self.pre_physics_step_none()

        if self.actor_init_pos == "back_to_back":
            a = self.modified_ref_body_pos[
                :, self.num_bodies : self.num_bodies + self.num_bodies, :
            ]
            b = self.modified_rb_body_pos[
                :, self.num_bodies : self.num_bodies + self.num_bodies, :
            ]

            angle = self.new_angles[
                :, 0
            ]  # self.anim_root_rot_yaw_cache[ (num_agents * ii)] + np.pi - self.anim_root_rot_yaw_cache[1 + (num_agents * ii)]
            rotation_axis = (
                torch.tensor([0.0, 0.0, 1.0]).repeat(self.num_envs, 1).to(self.device)
            )
            # print(angle)
            for kk in range(24):
                b[:, kk] = self.rotate_vector(b[:, kk], rotation_axis, angle)
                a[:, kk] = self.rotate_vector(a[:, kk], rotation_axis, angle)

        self.modified_ref_body_pos[..., :2] += self.additive_agent_pos[..., :2]
        self.modified_rb_body_pos[..., :2] += self.additive_agent_pos[..., :2]

        # allow changes in root using policy
        # print(full_action.reshape(self.num_envs, num_agents, -1)[...,np.newaxis,:].repeat(1,1,24,1).shape)

        # if(self.root_motion):
        #     self.modified_ref_body_pos.reshape(self.num_envs * num_agents,-1, 3)[...,:2] +=cached_root_pos
        #     self.modified_rb_body_pos.reshape(self.num_envs * num_agents,-1, 3)[...,:2] +=  cached_root_pos

        self.modified_ref_body_pos_no_physics = self.modified_ref_body_pos
        # modified_ref_body_root_pos = self.modified_ref_body_pos_no_physics.reshape(self.num_envs,num_agents,-1,3)
        # modified_ref_body_root_pos[...,:2] += full_action.reshape(self.num_envs, num_agents, -1)[...,np.newaxis,:].repeat(1,1,24,1)
        self.obs_root_pos = self.modified_rb_body_pos.reshape(
            self.num_envs * num_agents, -1, 3
        ).clone()[:, 0, :]

        self.visualized_ref_body_pos = self.modified_ref_body_pos_no_physics.clone()
        self.visualized_rb_body_pos = self.modified_rb_body_pos.clone()
        if (
            self.num_envs <= 15
        ):  # visualized only when we have small num of environment (mostly used when we want to evaluate)


            if self.num_envs == 1:
                self.vid_visualized_rb_body_pos = self.visualized_rb_body_pos.clone()
                self.vid_visualized_ref_body_pos = self.visualized_ref_body_pos.clone()

                # making sure video related visualization is centered around the origin.
                self.vid_visualized_rb_body_pos[..., :2] -= self.additive_agent_pos[
                    ..., :2
                ]
                self.vid_visualized_rb_body_pos = (
                    self.vid_visualized_rb_body_pos.reshape(1, 2, -1, 3)
                )
                self.vid_visualized_rb_body_pos[:, 1, ...] += (
                    self.additive_agent_pos.reshape(1, 2, -1, 3)[:, 1, ...]
                    - self.additive_agent_pos.reshape(1, 2, -1, 3)[:, 0, ...]
                )
                self.vid_visualized_rb_body_pos = (
                    self.vid_visualized_rb_body_pos.reshape(1, -1, 3)
                )

                self.vid_visualized_ref_body_pos[..., :2] -= self.additive_agent_pos[
                    ..., :2
                ]
                self.vid_visualized_ref_body_pos = (
                    self.vid_visualized_ref_body_pos.reshape(1, 2, -1, 3)
                )
                self.vid_visualized_ref_body_pos[:, 1, ...] += (
                    self.additive_agent_pos.reshape(1, 2, -1, 3)[:, 1, ...]
                    - self.additive_agent_pos.reshape(1, 2, -1, 3)[:, 0, ...]
                )
                self.vid_visualized_ref_body_pos = (
                    self.vid_visualized_ref_body_pos.reshape(1, -1, 3)
                )

        self._marker_pos[:] = torch.cat(
            [self.visualized_ref_body_pos, self.visualized_rb_body_pos], dim=1
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._marker_actor_ids),
            len(self._marker_actor_ids),
        )

        # NOTE: In Nam Hee's code this is set to zero
        # self.ref_body_vel = ref_body_vel #0.5 * (self._rigid_body_vel.reshape(-1,24,3) + ref_body_vel)

        self.step_count += 1
        self.progress_buf += 1

        return

    def pre_physics_step_ae(
        self,
        input_lats_importance=1e0,
        input_my_lats_importance=1e0,
        force_t_pose=False,
    ):

        self.my_lats = self.ae.encoder.forward(
            self.ae.rms.normalize(self.ref_rb_pos.reshape(self.ref_rb_pos.shape[0], -1))
        )

        sum_lats = (
            input_lats_importance * self.input_lats
            + input_my_lats_importance * self.my_lats
        )

        self.ref_body_pos = self.ae.decoder.forward(sum_lats)

        if force_t_pose:

            # Override to ensure the second character stays in the T-pose.
            self.ref_body_pos = self.ref_body_pos.reshape(
                self.num_envs, num_agents, -1, 3
            )
            self.ref_body_pos[:, 1, ...] = self.ref_rb_pos.reshape(
                self.num_envs, num_agents, -1, 3
            )[:, 1, ...].clone()

        self.ref_body_pos = self.ref_body_pos.reshape(self.num_envs, -1, 3)

        self.modified_ref_body_pos = self.ref_body_pos.clone()
        self.modified_rb_body_pos = self.ref_rb_pos.reshape(
            self.num_envs, -1, 3
        ).clone()

    def pre_physics_step_vae(
        self,
        input_lats_importance=1e0,
        input_my_lats_importance=1e0,
        force_t_pose=False,
    ):

        self.my_lats = self.ae.encoder.forward(
            self.ae.rms.normalize(self.ref_rb_pos.reshape(self.ref_rb_pos.shape[0], -1))
        )
        mu = self.ae.mu(self.my_lats)

        sum_lats = (
            input_lats_importance * self.input_lats + input_my_lats_importance * mu
        )

        self.ref_body_pos = self.ae.decoder.forward(sum_lats)

        if force_t_pose:

            # Override to ensure the second character stays in the T-pose.
            self.ref_body_pos = self.ref_body_pos.reshape(
                self.num_envs, num_agents, -1, 3
            )
            self.ref_body_pos[:, 1, ...] = self.ref_rb_pos.reshape(
                self.num_envs, num_agents, -1, 3
            )[:, 1, ...].clone()

        self.ref_body_pos = self.ref_body_pos.reshape(self.num_envs, -1, 3)

        self.modified_ref_body_pos = self.ref_body_pos.clone()
        self.modified_rb_body_pos = self.ref_rb_pos.reshape(
            self.num_envs, -1, 3
        ).clone()

    def pre_physics_step_cvae(
        self,
        motion_res,
        input_lats_importance=1e0,
        input_my_lats_importance=1e0,
        force_t_pose=False,
    ):
        # motion_res["rb_rot"]
        # make data ready for CVAE
        rb_rot_sixd = torch.as_tensor(
            Rotation.from_quat(motion_res["rb_rot"].reshape(-1, 4))
            .as_matrix()
            .reshape((*motion_res["rb_rot"].shape[:-1], 3, 3))[..., :2]
            .reshape((*motion_res["rb_rot"].shape[:-1], 6)),
            dtype=torch.float,
            device=self.device,
        )
        rb_rot_sixd_inv, root_yaw = remove_root_yaw_from_sixd(rb_rot_sixd)
        xs = rb_rot_sixd_inv.reshape(-1, 24, 6)[:, 1:]
        # xs = torch.tensor(xs, dtype=torch.float, device=self.device)
        xs = xs.reshape(xs.shape[0], -1)

        # raw_ys for root position and orientation that ignores any kind of invariance snapping
        raw_ys = torch.cat(
            [
                motion_res["rg_pos"].reshape(-1, 24, 3)[:, 0],
                rb_rot_sixd.reshape(-1, 24, 6)[:, 0, :],
            ],
            dim=-1,
        )  # original yaw and xy included

        # encoder_ys has same information as raw_ys except we apply invariance, i.e. we remove root xy position and remove yaw
        encoder_ys = torch.cat(
            [
                motion_res["rg_pos"].reshape(-1, 24, 3)[:, 0, [-1]],  # z position
                rb_rot_sixd_inv.reshape(-1, 24, 6)[:, 0, :],  # yaw-less orientation
            ],
            dim=-1,
        )
        
        self.blue_rb_root_xyz = motion_res["rg_pos"][:, 0]
        self.blue_rb_root_rot_sixd = rb_rot_sixd[:, 0]
        self.blue_rb_root_vel = motion_res["body_vel"][:, 0]
        self.blue_rb_root_ang_vel = motion_res["body_ang_vel"][:, 0]
        self.blue_sixd = rb_rot_sixd.reshape(self.num_envs * num_agents, -1)
        self.blue_joint_ang_vel = motion_res["body_ang_vel"].reshape(
            self.num_envs * num_agents, -1
        )

        _, decoded, mu, log_var = self.ae.forward(
            xs,
            encoder_ys,
            train_yes=False,
        )

        self.xyz_edit = self.input_lats[..., :3] * 1
        # self.xyz_edit[..., 1:] = 0
        # print(xyz_edit)
        rpy_edit = self.input_lats[..., 3:6] * torch.pi
        latent_edit = self.input_lats[..., 6:]

        # For sanity check, just apply xyz and latent. Worry about rpy later.
        # edited_ys = torch.cat([self.ref_body_root_pos, self.ref_body_root_rot], dim=-1)
        edited_ys = raw_ys * 1
        # change the root xy based on action
        # TODO maybe instead of doing this, it would be good to condition on root xy as well. not exactly but some form of it.
        # print(self.xyz_edit[..., :2])
        edited_ys[..., :2] += self.xyz_edit[..., :2] * 0
        edited_rotmat = sixd_to_rotmat(edited_ys[..., 3:9])
        yaw_edit = rpy_edit[..., -1] * 0
        edit_rot = Rotation.from_euler("Z", yaw_edit)
        edited_rot = edit_rot * Rotation.from_matrix(edited_rotmat)

        z = input_lats_importance * latent_edit + input_my_lats_importance * mu

        decoder_ys = encoder_ys * 1  # NOTE: later we want to include edits to root here

        # # change root z of decoder using actions
        # decoder_ys_root_z = decoder_ys[..., 0]
        # decoder_ys_root_z += xyz_edit[..., 2]

        # # change pitch and roll of root based on action (we can fully rotate and then snap yaw, or just rotate pitch and roll?)
        # decoder_ys_rotmat = sixd_to_rotmat(decoder_ys[..., 1:7])
        # decoder_pitch_roll_edit = rpy_edit[..., 0:3] * 1
        # decoder_pitch_roll_edit[..., 0] = 0
        # decoder_pitch_roll_edit_rot = Rotation.from_euler(
        #     "zyx", decoder_pitch_roll_edit
        # )
        # decoder_edited_rot = decoder_pitch_roll_edit_rot * Rotation.from_matrix(
        #     decoder_ys_rotmat
        # )

        # decoder_good_sixd = torch.as_tensor(
        #     decoder_edited_rot.as_matrix().reshape((-1, 3, 3))[..., :2].reshape(-1, 6),
        #     dtype=torch.float,
        #     device=self.device,
        # )

        # decoder_ys[..., 1:7] = decoder_good_sixd

        cvae_decoded = self.ae.decode(z, decoder_ys)
        cvae_decoded = cvae_decoded.reshape(self.num_envs * num_agents, 24, -1)

        # The goal is to rotate the rotationally invariant CVAE output by the specified information in edit_ys
        # Note that we only need to add back yaw because pitch and roll are taken into account by the decoder
        # edited_rotmat = sixd_to_rotmat(edited_ys[..., 3:9])
        # edited_rot = Rotation.from_matrix(edited_rotmat)

        edited_yaw = edited_rot.as_euler("ZYX")[..., 0]
        good_rotmat = torch.as_tensor(
            Rotation.from_euler("Z", edited_yaw).as_matrix(),
            dtype=torch.float,
            device=self.device,
        )
        good_sixd = torch.as_tensor(
            edited_rot.as_matrix().reshape((-1, 3, 3))[..., :2].reshape(-1, 6),
            dtype=torch.float,
            device=self.device,
        )

        # Apply rotation
        # 1. Remove root from position so it is positioned at the origin
        tmp = cvae_decoded[:, [0]] * 1
        cvae_decoded -= tmp

        # 2. rotate yaw
        cvae_decoded_with_yaw = torch.einsum(
            "nab, npb -> npa", good_rotmat, cvae_decoded
        )

        # 3. add back the position
        cvae_decoded_with_yaw += tmp

        # Bring to original position
        cvae_decoded_with_yaw[..., :2] += edited_ys[..., None, :2]
        cvae_decoded_with_yaw[..., -1] -= tmp[..., -1]  # TODO WHY?
        cvae_decoded_with_yaw[..., -1] += edited_ys[..., None, 2]

        #NOTE this is wrong, remove the num_agents
        
        self.blue_rb_xyz = motion_res["rg_pos"].reshape(self.num_envs * num_agents, -1)
        self.blue_rb_rot = motion_res["rb_rot"].reshape(self.num_envs * num_agents, -1)
        self.ref_body_pos = cvae_decoded_with_yaw

        # Use these as stateful features
        self.ref_body_root_pos = edited_ys[
            ..., :3
        ]  # TODO this is not used anywhjere but isn't the actual root pose cvae_decoded_with_yaw[..., 0, :3] ? does it matter?
        self.ref_body_root_rot = good_sixd
        if force_t_pose:

            # Override to ensure the second character stays in the T-pose.
            self.ref_body_pos = self.ref_body_pos.reshape(
                self.num_envs, num_agents, -1, 3
            )
            self.ref_body_pos[:, 1, ...] = self.ref_rb_pos.reshape(
                self.num_envs, num_agents, -1, 3
            )[:, 1, ...].clone()

        self.prev_red_rb_root_xyz = self.red_rb_root_xyz.clone()
        self.red_rb_root_xyz = cvae_decoded_with_yaw[..., 0, :3]  # edited_ys[..., :3]
        self.prev_red_rb_xyz = self.red_rb_xyz * 1
        self.red_rb_xyz = self.ref_body_pos.reshape(self.num_envs * num_agents, -1)
        self.ref_body_pos = self.ref_body_pos.reshape(self.num_envs, -1, 3)
        self.modified_ref_body_pos = self.ref_body_pos.clone()

        # self._body_names_orig = ["Pelvis","L_Hip","L_Knee","L_Ankle","L_Toe","R_Hip","R_Knee","R_Ankle","R_Toe","Torso","Spine","Chest","Neck","Head","L_Thorax","L_Shoulder","L_Elbow","L_Wrist","L_Hand","R_Thorax","R_Shoulder","R_Elbow","R_Wrist","R_Hand",]

        right_shoulder_index = self._body_names.index("R_Thorax")
        right_shoulder_rot_sixd = rb_rot_sixd_inv.reshape(-1, 24, 6)[
            :, right_shoulder_index, :
        ]
        right_shoulder_rotmat = sixd_to_rotmat(right_shoulder_rot_sixd)
        rs_edit_rot = Rotation.from_euler("Y", 130)
        rs_edited_rot = rs_edit_rot * Rotation.from_matrix(right_shoulder_rotmat)
        rs_edited_yaw = rs_edited_rot.as_euler("ZYX")[..., 1]
        rs_good_rotmat = torch.as_tensor(
            Rotation.from_euler("Y", rs_edited_yaw).as_matrix(),
            dtype=torch.float,
            device=self.device,
        )

        # Apply rotation
        # 1. Remove root from position so it is positioned at the origin
        new_ref_rb_pos = self.ref_rb_pos.reshape(self.num_envs, -1, 3).clone()
        chain_list_indecies = []
        chain_list = ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]
        for i in chain_list:
            chain_list_indecies.append(self._body_names.index(i))
        rs_ref_rb_pos = new_ref_rb_pos[:, chain_list_indecies]
        tmp = rs_ref_rb_pos[:, [0]] * 1
        rs_ref_rb_pos -= tmp

        # 2. rotate yaw
        new_rs_ref_rb_pos = torch.einsum(
            "nab, npb -> npa", rs_good_rotmat, rs_ref_rb_pos
        )

        # 3. add back the position
        new_rs_ref_rb_pos += tmp

        new_ref_rb_pos[:, chain_list_indecies] = new_rs_ref_rb_pos
        self.blue_roteated_rb_pos = new_ref_rb_pos.reshape(self.num_envs, -1, 3).clone()
        self.modified_rb_body_pos = new_ref_rb_pos.reshape(self.num_envs, -1, 3).clone()
    
    def pre_physics_step_pvae(
        self,
        motion_res,
        input_lats_importance=1e0,
        input_my_lats_importance=1e0,
    ):

        # make data ready for PVAE
        rb_rot_sixd = torch.as_tensor(
            Rotation.from_quat(motion_res["rb_rot"].reshape(-1, 4))
            .as_matrix()
            .reshape((*motion_res["rb_rot"].shape[:-1], 3, 3))[..., :2]
            .reshape((*motion_res["rb_rot"].shape[:-1], 6)),
            dtype=torch.float,
            device=self.device,
        )
        rb_rot_sixd_inv, root_yaw = remove_root_yaw_from_sixd(rb_rot_sixd)
        self.blue_sixd_inv = rb_rot_sixd_inv.reshape(-1,144) 
        xs = rb_rot_sixd_inv
        # xs = torch.tensor(xs, dtype=torch.float, device=self.device)
        xs = xs.reshape(xs.shape[0],24, -1)

        # raw_ys for root position and orientation that ignores any kind of invariance snapping
        raw_ys = torch.cat(
            [
                motion_res["rg_pos"].reshape(-1, 24, 3)[:, 0],
                rb_rot_sixd.reshape(-1, 24, 6)[:, 0, :],
            ],
            dim=-1,
        )  # original yaw and xy included

        self.prev_blue_rb_root_xyz = self.blue_rb_root_xyz * 1
        self.blue_rb_root_xyz = motion_res["rg_pos"][:, 0] * 1
        self.blue_rb_root_rot_sixd = rb_rot_sixd[:, 0]
        self.blue_rb_root_vel = motion_res["body_vel"][:, 0]
        self.blue_rb_root_ang_vel = motion_res["body_ang_vel"][:, 0]
        self.blue_sixd = rb_rot_sixd.reshape(self.num_envs * num_agents, -1)
        self.blue_joint_ang_vel = motion_res["body_ang_vel"].reshape(
            self.num_envs * num_agents, -1
        )

        _, decoded, mu, log_var = self.ae.forward(
            xs,
            train_yes=False,
        )
        self.blue_latent = mu * 1
        # self.xyz_edit = self.input_lats[..., :3] * 1
        # rpy_edit = self.input_lats[..., 3:6] * torch.pi
        # latent_edit = self.input_lats[..., 6:]
        latent_edit = self.input_lats[..., :latent_dim]
        if(self.gate_pvae):
            gates = self.input_lats[..., -5:]
            
            gates = (torch.abs(gates) > 0.5).float()
            print(gates[0])
            gates = torch.repeat_interleave(gates, torch.tensor([4,3,4,4,4], device = self.device), dim = -1)
            latent_edit *= gates
            #print(latent_edit[0])

        edited_ys = raw_ys * 1
        # edited_ys[..., :2] += self.xyz_edit[..., :2] * 0
        # edited_rotmat = sixd_to_rotmat(edited_ys[..., 3:9])
        # yaw_edit = rpy_edit[..., -1] * 0
        # edit_rot = Rotation.from_euler("Z", yaw_edit)
        # edited_rot = edit_rot * Rotation.from_matrix(edited_rotmat)

        # it is 3x because we assume that the latent is a normal distribution so the latent should be between -3 and -3
        z = 1 * input_lats_importance * latent_edit + input_my_lats_importance * mu

        self.red_latent = z * 1
        #z[..., 6:] =  input_lats_importance * latent_edit[..., 6:]
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.ae.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.ae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        cvae_decoded = cvae_decoded.reshape(-1, 144)
        self.red_sixd_inv = cvae_decoded * 1
        recon_rot_sixd_reshaped = add_root_yaw_to_sixd(cvae_decoded, root_yaw).reshape(-1,24,3,2)
        self.red_sixd = torch.as_tensor(recon_rot_sixd_reshaped.reshape(-1,144) * 1, device=self.device, dtype=torch.float, )

        third_column = np.cross(
            recon_rot_sixd_reshaped[..., 0],
            recon_rot_sixd_reshaped[..., 1],
            axis=-1,
        )
        recon_rot_rotmat = np.concatenate(
            [recon_rot_sixd_reshaped, third_column[..., None]], axis=-1
        )
        recon_rot_quat = Rotation.from_matrix(recon_rot_rotmat.reshape(-1,3,3)).as_quat()
        recon_rot_quat = recon_rot_quat.reshape(-1,24,4)
        recon_rot_quat =  torch.as_tensor(recon_rot_quat, dtype=torch.float, device = self.device)

        # if(self.physics_enable):
        #     current_root = self._rigid_body_pos.reshape(-1,24,3)[:,0]
        # else:
        current_root = self.red_rb_root_xyz
        changed_root = current_root * 1
        
        if(self.gate_pvae):
            changed_root[..., :2] += self.input_lats[...,-7:-5] 
        else:
            changed_root[..., :2] += self.input_lats[...,-2:] 

        if(self.root_motion):
            changed_root = motion_res["root_pos"]

        edited_ys[..., :3] = changed_root * 1

        recon_sk_state = SkeletonState.from_rotation_and_root_translation(
            self.sk_tree,
            torch.as_tensor(recon_rot_quat, dtype=torch.float),
            torch.as_tensor(edited_ys[..., :3], dtype=torch.float),
            is_local=False,
        )

        cvae_decoded_with_yaw = recon_sk_state.global_translation.reshape(-1,24,3)
        cvae_decoded_with_yaw = cvae_decoded_with_yaw.to(self.device)

        self.prev_blue_rb_xyz = self.blue_rb_xyz
        self.blue_rb_xyz = motion_res["rg_pos"].reshape(self.num_envs * num_agents, -1)
        self.blue_rb_rot = motion_res["rb_rot"].reshape(self.num_envs * num_agents, -1)
        self.ref_body_pos = cvae_decoded_with_yaw

        # Use these as stateful features
        self.ref_body_root_pos = edited_ys[
            ..., :3
        ]  # TODO this is not used anywhjere but isn't the actual root pose cvae_decoded_with_yaw[..., 0, :3] ? does it matter?
        #self.ref_body_root_rot = good_sixd


        self.prev_red_rb_root_xyz = self.red_rb_root_xyz.clone()
        self.red_rb_root_xyz = cvae_decoded_with_yaw[..., 0, :3]  # edited_ys[..., :3]
        self.prev_red_rb_xyz = self.red_rb_xyz * 1
        self.red_rb_xyz = self.ref_body_pos.reshape(self.num_envs * num_agents, -1)
        self.ref_body_pos = self.ref_body_pos.reshape(self.num_envs, -1, 3)
        self.modified_ref_body_pos = self.ref_body_pos.clone()

        # # self._body_names_orig = ["Pelvis","L_Hip","L_Knee","L_Ankle","L_Toe","R_Hip","R_Knee","R_Ankle","R_Toe","Torso","Spine","Chest","Neck","Head","L_Thorax","L_Shoulder","L_Elbow","L_Wrist","L_Hand","R_Thorax","R_Shoulder","R_Elbow","R_Wrist","R_Hand",]

        # right_shoulder_index = self._body_names.index("R_Thorax")
        # right_shoulder_rot_sixd = rb_rot_sixd_inv.reshape(-1, 24, 6)[
        #     :, right_shoulder_index, :
        # ]
        # right_shoulder_rotmat = sixd_to_rotmat(right_shoulder_rot_sixd)
        # rs_edit_rot = Rotation.from_euler("Y", 130)
        # rs_edited_rot = rs_edit_rot * Rotation.from_matrix(right_shoulder_rotmat)
        # rs_edited_yaw = rs_edited_rot.as_euler("ZYX")[..., 1]
        # rs_good_rotmat = torch.as_tensor(
        #     Rotation.from_euler("Y", rs_edited_yaw).as_matrix(),
        #     dtype=torch.float,
        #     device=self.device,
        # )

        # # Apply rotation
        # # 1. Remove root from position so it is positioned at the origin
        # new_ref_rb_pos = self.ref_rb_pos.reshape(self.num_envs, -1, 3).clone()
        # chain_list_indecies = []
        # chain_list = ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]
        # for i in chain_list:
        #     chain_list_indecies.append(self._body_names.index(i))
        # rs_ref_rb_pos = new_ref_rb_pos[:, chain_list_indecies]
        # tmp = rs_ref_rb_pos[:, [0]] * 1
        # rs_ref_rb_pos -= tmp

        # # 2. rotate yaw
        # new_rs_ref_rb_pos = torch.einsum(
        #     "nab, npb -> npa", rs_good_rotmat, rs_ref_rb_pos
        # )

        # # 3. add back the position
        # new_rs_ref_rb_pos += tmp

        # new_ref_rb_pos[:, chain_list_indecies] = new_rs_ref_rb_pos
        #self.blue_roteated_rb_pos = new_ref_rb_pos.reshape(self.num_envs, -1, 3).clone()
        self.modified_rb_body_pos = self.ref_rb_pos.reshape(
            self.num_envs, -1, 3
        ).clone()
    def pre_physics_step_pvae_dof(
        self,
        motion_res,
        input_lats_importance=1e0,
        input_my_lats_importance=1e0,
    ):


        xs = motion_res["dof_pos"]
        # xs = torch.tensor(xs, dtype=torch.float, device=self.device)
        xs = xs.reshape(xs.shape[0],23, -1)

        # raw_ys for root position and orientation that ignores any kind of invariance snapping
        raw_ys = torch.cat(
            [
                motion_res["rg_pos"].reshape(-1, 24, 3)[:, 0],
            ],
            dim=-1,
        )  # original yaw and xy included

        self.prev_blue_rb_root_xyz = self.blue_rb_root_xyz * 1
        self.blue_rb_root_xyz = motion_res["rg_pos"][:, 0] * 1
        self.blue_rb_root_vel = motion_res["body_vel"][:, 0]
        self.blue_rb_root_ang_vel = motion_res["body_ang_vel"][:, 0]
        self.blue_joint_ang_vel = motion_res["body_ang_vel"].reshape(
            self.num_envs * num_agents, -1
        )

        _, decoded, mu, log_var = self.ae.forward(
            xs,
            train_yes=False,
        )
        self.blue_latent = mu * 1
        
        latent_edit = self.input_lats[..., :latent_dim]
        if(self.gate_pvae):
            gates = self.input_lats[..., -5:]
            
            gates = (torch.abs(gates) > 0.5).float()
            gates = torch.repeat_interleave(gates, torch.tensor([4,3,4,4,4], device = self.device), dim = -1)
            latent_edit *= gates

        edited_ys = raw_ys * 1
        #z = mu * 1
        #z[..., 11:] += 10 * latent_edit[..., 11:]
        # it is 3x because we assume that the latent is a normal distribution so the latent should be between -3 and -3
        coef = [ 1.        ,  1.39494009,  1.94585785,  2.71435512,  3.78636278,
        5.28174923,  7.36772374, 10.27753321, 14.33654309, 19.9986187 ]
        #NOTE ignores lower body
        latent_edit[..., :11] = latent_edit[..., :11] * 0

        #print(self.sweep1)
        z = coef[3] * input_lats_importance * latent_edit + input_my_lats_importance * mu
        
        self.red_latent = z * 1
        #z[..., 6:] =  input_lats_importance * latent_edit[..., 6:]
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.ae.hammer_size):
           
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.ae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        cvae_decoded = cvae_decoded.reshape((1, 23, -1))

        pose_quat = Rotation.from_rotvec(cvae_decoded.reshape(-1, 3)).as_quat().reshape(-1, 23, 4)
        root_rot = motion_res["root_rot"]

        
        if(self.calc_root_dir):
            heading_direction = self.red_rb_root_xyz - self.prev_red_rb_root_xyz
            heading_vectors = heading_direction/ heading_direction.norm(dim=-1, keepdim=True)
            # Compute the yaw angles (arctan2(y, x) gives the angle in the xy-plane)
            
            root_yaw =  torch.tensor(Rotation.from_quat(root_rot).as_euler('zyx')[...,[0]], dtype=torch.float, device=self.device)
            self.prev_red_yaw_angles = self.current_red_yaw_angles
            if(self.root_dir_action):
               # self.current_red_yaw_angles =  root_yaw + self.input_lats[...,latent_dim+2:] 
                self.current_red_yaw_angles +=  self.input_lats[...,latent_dim+2:] 
            else:
                self.current_red_yaw_angles = torch.atan2(heading_vectors[:, 1], heading_vectors[:, 0]).reshape(-1,1)

            nan_mask = torch.isnan(self.current_red_yaw_angles)
            nan_rows = nan_mask.any(dim=1)
            self.current_red_yaw_angles[nan_rows] = root_yaw[nan_rows]
            
            # Create the rotation matrices and quaternions using only the yaw angles
            rotations = Rotation.from_euler('z', self.current_red_yaw_angles.numpy())

            quaternions = torch.tensor(rotations.as_quat(), dtype=torch.float, device=self.device)
            
            
            pose_quat = np.concatenate([quaternions[:, None, :], pose_quat], axis=1)
        else:
            pose_quat = np.concatenate([root_rot[:, None, :], pose_quat], axis=1)

        current_root = self.red_rb_root_xyz
        changed_root = current_root * 1
        
        if(self.calc_root_dir):
            x_components = torch.cos( self.current_red_yaw_angles )
            y_components = torch.sin( self.current_red_yaw_angles )
    
            direction_vectors = torch.concatenate((x_components, y_components), dim=-1)
            # direction_vectors[..., 0] = 1
            # direction_vectors[...,1] = 0
            c = [0.01      , 0.01668101, 0.02782559, 0.04641589, 0.07742637, 0.12915497, 0.21544347, 0.35938137, 0.59948425, 1.        ]

            self.input_lats[...,latent_dim:latent_dim+2] = c[9] * torch.abs(self.input_lats[...,latent_dim:latent_dim+2]) * direction_vectors


        
        if(self.gate_pvae):
            #Note not working if we have root_dir_action
            changed_root[..., :2] += self.input_lats[...,-7:-5] 
        elif (not self.root_motion):
            changed_root[..., :2] += self.input_lats[...,latent_dim:latent_dim+2] 

        if(self.root_motion):
            changed_root = motion_res["root_pos"]

        edited_ys[..., :3] = changed_root * 1
        recon_sk_state = SkeletonState.from_rotation_and_root_translation(
            self.sk_tree,
            torch.as_tensor(pose_quat,dtype=torch.float).cpu(),
            torch.as_tensor(edited_ys[..., :3],dtype=torch.float).cpu(),
            is_local=True
        )

        cvae_decoded_with_yaw = recon_sk_state.global_translation.reshape(-1,24,3)
        cvae_decoded_with_yaw = cvae_decoded_with_yaw.to(self.device)

        self.prev_blue_rb_xyz = self.blue_rb_xyz
        self.blue_rb_xyz = motion_res["rg_pos"].reshape(self.num_envs * num_agents, -1)
        self.blue_rb_rot = motion_res["rb_rot"].reshape(self.num_envs * num_agents, -1)
        self.ref_body_pos = cvae_decoded_with_yaw

        # Use these as stateful features
        self.ref_body_root_pos = edited_ys[
            ..., :3
        ]  # TODO this is not used anywhjere but isn't the actual root pose cvae_decoded_with_yaw[..., 0, :3] ? does it matter?
        #self.ref_body_root_rot = good_sixd


        self.prev_red_rb_root_xyz = self.red_rb_root_xyz.clone()
        self.red_rb_root_xyz = cvae_decoded_with_yaw[..., 0, :3]  # edited_ys[..., :3]
        self.prev_red_rb_xyz = self.red_rb_xyz * 1
        self.red_rb_xyz = self.ref_body_pos.reshape(self.num_envs * num_agents, -1)
        self.ref_body_pos = self.ref_body_pos.reshape(self.num_envs, -1, 3)
        self.modified_ref_body_pos = self.ref_body_pos.clone()

        self.modified_rb_body_pos = self.ref_rb_pos.reshape(
            self.num_envs, -1, 3
        ).clone()

    def pre_physics_step_dof(
        self,
        motion_res,
        input_lats_importance=1e0,
        input_my_lats_importance=1e0,
    ):
        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            smpl_params,
            limb_weights,
            pose_aa,
            ref_rb_pos,
            ref_rb_rot,
            ref_body_vel,
            ref_body_ang_vel,
            ) = (
                motion_res["root_pos"],
                motion_res["root_rot"],
                motion_res["dof_pos"],
                motion_res["root_vel"],
                motion_res["root_ang_vel"],
                motion_res["dof_vel"],
                motion_res["motion_bodies"],
                motion_res["motion_limb_weights"],
                motion_res["motion_aa"],
                motion_res["rg_pos"],
                motion_res["rb_rot"],
                motion_res["body_vel"],
                motion_res["body_ang_vel"],
            )

        self.prev_blue_rb_root_xyz = self.blue_rb_root_xyz * 1
        self.blue_rb_root_xyz = motion_res["rg_pos"][:, 0] * 1
        self.prev_blue_rb_xyz = self.blue_rb_xyz
        self.blue_rb_xyz = motion_res["rg_pos"].reshape(self.num_envs * num_agents, -1) * 1

        dof_pos = motion_res["dof_pos"]
        if(self.physics_enable):
            current_root = self._rigid_body_pos.reshape(-1,24,3)[:,0]
            changed_root = current_root * 1
            changed_root[..., :2] += self.input_lats[...,69:] 
        elif not self.root_motion:
            current_root = self.red_rb_root_xyz
        #print(current_root)
            changed_root = current_root * 1
            changed_root[..., :2] += self.input_lats[...,69:] 
        current_root = self.red_rb_root_xyz
        #print(current_root)
        changed_root = current_root * 1
        changed_root[..., :2] += self.input_lats[...,69:] 

        if(self.root_motion):
            changed_root = motion_res["root_pos"]

        self.prev_red_rb_root_xyz = self.red_rb_root_xyz
        self.red_rb_root_xyz = changed_root
        # print(self.input_lats[0,:69])
        # print(self.input_lats.shape)
        # print(self.input_lats[0,:69].shape)
        #self.blue_dof_pos = dof_pos * 1
        #for testing, lets set the lower body to 0
        self.input_lats[...,:24] = self.input_lats[...,:24] * 0
        dof_pos += self.input_lats[...,:69] * 1000
        #print(dof_pos)
        #self.red_dof_pos = dof_pos * 1 

        pose_quat = Rotation.from_rotvec(dof_pos.reshape(-1, 3)).as_quat().reshape(-1, 23, 4)
        root_rot = motion_res["root_rot"]

        pose_quat = np.concatenate([root_rot[:, None, :], pose_quat], axis=1)
        
        recon_sk_state = SkeletonState.from_rotation_and_root_translation(
            self.sk_tree,
            torch.as_tensor(pose_quat,dtype=torch.float).cpu(),
            torch.as_tensor(changed_root,dtype=torch.float).cpu(),
            is_local=True
        )
        recon_global_pos = recon_sk_state.global_translation.reshape(-1,24,3)
        recon_global_pos = recon_global_pos.to(self.device)


        #self.ref_body_pos = self.ref_rb_pos.reshape(self.num_envs, -1, 3).clone()
        self.modified_ref_body_pos = recon_global_pos.reshape(self.num_envs, -1, 3).clone()
        self.prev_red_rb_xyz = self.red_rb_xyz * 1
        self.red_rb_xyz = self.modified_ref_body_pos.reshape(self.num_envs, -1) * 1

        self.modified_rb_body_pos = motion_res["rg_pos"].reshape(self.num_envs, -1, 3).clone()

    def pre_physics_step_none(self):
        self.ref_body_pos = self.ref_rb_pos.reshape(self.num_envs, -1, 3).clone()
        self.modified_ref_body_pos = self.ref_rb_pos.reshape(
            self.num_envs, -1, 3
        ).clone()
        self.modified_rb_body_pos = self.ref_rb_pos.reshape(
            self.num_envs, -1, 3
        ).clone()

    def sixd_to_euler(self, rb_rot_sixd):
        rb_rot_sixd_reshaped = rb_rot_sixd.reshape(-1, 3, 2)
        third_column = np.cross(
            rb_rot_sixd_reshaped[..., 0], rb_rot_sixd_reshaped[..., 1], axis=-1
        )
        rb_rot_rotmat = np.concatenate(
            [rb_rot_sixd_reshaped, third_column[..., None]], axis=-1
        )
        rb_rot_euler = Rotation.from_matrix(rb_rot_rotmat.reshape(-1, 3, 3)).as_euler(
            "zyx"
        )

        return rb_rot_euler

    def sixd_add_root(self, rb_rot_sixd, euler_orientation):

        rb_rot_sixd_reshaped = rb_rot_sixd.reshape(self.num_envs * num_agents, -1, 3, 2)
        euler_orientation = euler_orientation.reshape(self.num_envs * num_agents, -1, 3)
        third_column = np.cross(
            rb_rot_sixd_reshaped[..., 0], rb_rot_sixd_reshaped[..., 1], axis=-1
        )
        rb_rot_rotmat = np.concatenate(
            [rb_rot_sixd_reshaped, third_column[..., None]], axis=-1
        )
        rb_rot_euler = Rotation.from_matrix(rb_rot_rotmat.reshape(-1, 3, 3)).as_euler(
            "zyx"
        )

        rb_rot_euler.reshape(*rb_rot_sixd_reshaped.shape[:-2], 3)[
            :, :, :
        ] += euler_orientation
        new_rb_rot_sixd = (
            Rotation.from_euler("zyx", rb_rot_euler)
            .as_matrix()
            .reshape((*rb_rot_sixd_reshaped.shape[:-2], 3, 3))[..., :2]
            .reshape((*rb_rot_sixd_reshaped.shape[:-2], 6))
        )
        return new_rb_rot_sixd

    def rotate_vector(self, vector, axis, angle):

        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)

        cross_product = torch.linalg.cross(axis, vector)
        dot_product = torch.sum(axis * vector, dim=-1).unsqueeze(-1)

        rotated_vector = (
            vector * cos_theta
            + cross_product * sin_theta
            + axis * dot_product * (1 - cos_theta)
        )

        return rotated_vector

    def quaternion_to_direction_vector(self, quaternion):

        q0, q1, q2, q3 = quaternion

        # Compute direction vector components
        x = 2 * (q1 * q3 - q0 * q2)
        y = 2 * (q0 * q1 + q2 * q3)
        z = q0**2 - q1**2 - q2**2 + q3**2

        # Create and return the direction vector tensor
        direction_vector = torch.tensor(
            [x, y, z], dtype=quaternion.dtype, device=self.device
        )
        return direction_vector

    def yaw_to_direction_vector(self, yaw):
        # Compute direction vector components
        x = torch.cos(yaw)
        y = torch.sin(yaw)
        z = 0.0  # Assuming rotation around the vertical axis, so no change in the vertical component

        # Create and return the direction vector tensor
        direction_vector = torch.tensor(
            [x, y, z], dtype=torch.float32, device=self.device
        )
        return direction_vector

    def physics_step(self):
        for i in range(self.hlc_control_freq_inv):
            # self._rigid_body_pos.reshape(self.num_envs*num_agents, -1)
            # self._rigid_body_pos.reshape(self.num_envs*num_agents, self.num_bodies,3),
            obs = self._compute_observations(
                self._rigid_body_pos.reshape(
                    self.num_envs * num_agents, self.num_bodies, 3
                ),
                self._rigid_body_rot.reshape(
                    self.num_envs * num_agents,
                    self.num_bodies,
                    self._rigid_body_rot.shape[-1],
                ),
                self._rigid_body_vel.reshape(
                    self.num_envs * num_agents,
                    self.num_bodies,
                    self._rigid_body_vel.shape[-1],
                ),
                self._rigid_body_ang_vel.reshape(
                    self.num_envs * num_agents,
                    self.num_bodies,
                    self._rigid_body_ang_vel.shape[-1],
                ),
            )
            if(self.physics_enable):
                self.plot_scaler(
                "custom/ref_velocity",
                torch.mean(self.ref_body_vel.reshape(self.num_envs * num_agents, -1)),
                self.step_count,
            )
                #print(torch.mean(self.ref_body_vel.reshape(self.num_envs * num_agents, -1)))
                calc_speed = ((self.blue_rb_xyz - self.prev_blue_rb_xyz) / 0.01666666666).reshape(self.num_envs * num_agents, -1)
                #print(torch.mean(((self.red_rb_xyz - self.prev_red_rb_xyz) / 0.01666666666).reshape(self.num_envs * num_agents, -1)))
                self.plot_scaler(
                "custom/calc_velocity",
                torch.mean(calc_speed),
                self.step_count,
            )
                g = compute_imitation_observations_v7(
                self._humanoid_root_states.reshape(self.num_envs*num_agents,-1)[..., :3],
                    self._humanoid_root_states.reshape(self.num_envs*num_agents,-1)[..., 3:7],
                    self._rigid_body_pos.reshape(self.num_envs*num_agents, self.num_bodies,3),
                    self._rigid_body_vel.reshape(self.num_envs*num_agents, self.num_bodies,self._rigid_body_vel.shape[-1]),
                    self.modified_ref_body_pos, # TODO: this should be changed to use agent
                    #self.ref_body_vel.reshape(self.num_envs*num_agents, self.num_bodies,3),
                    ((self.red_rb_xyz - self.prev_red_rb_xyz) / 0.01666666666), 
                    1,
                    True
                )

                obs = obs[..., :358]
                # # TODO: We need both character's obs and both characters g, the we will have (2,*) tensors
                # # Also, the self.running_mean has to be repeated. we can do this, where it is assigned for the first time (after it is being read from pnn)

                nail = torch.cat([obs, g], dim=-1)
                nail = (nail - self.running_mean[None].float()) / torch.sqrt(
                    self.running_var[None].float() + 1e-05
                )
                nail = torch.clamp(nail, min=-5.0, max=5.0)
                _, x_all = self.pnn(nail)
                x_all = torch.stack(x_all, dim=1)
                # x_all = x_all.view(-1,x_all.shape[-2],x_all.shape[-1])
                # Compute the MCP policy's actions
                input_dict = {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": nail,  # .view(-1,nail.shape[-1]),
                    "rnn_states": None,
                }
                with torch.no_grad():
                    weights, _, _, _ = self.mcp(input_dict)
                rescaled_weights = rescale_actions(
                    -1.0, 1.0, torch.clamp(weights, -1.0, 1.0)
                )
                # rescaled_weights = rescaled_weights.view(self.num_envs, num_agents, -1)
                self.actions = torch.sum(rescaled_weights[:, :, None] * x_all, dim=1)
                self.actions = self.actions.view(self.num_envs, -1)

                if self.smpl_humanoid:
                    if self.reduce_action:
                        actions_full = torch.zeros(
                            [self.actions.shape[0], self._dof_size]
                        ).to(self.device)
                        actions_full[:, self.action_idx] = self.actions
                        pd_tar = self._action_to_pd_targets(actions_full)

                    else:
                        pd_tar = self._action_to_pd_targets(self.actions)
                pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
                self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
                #self.render()
                super().physics_step()
                self._refresh_sim_tensors()
            else:
                self.render()

            # shifting reward history to the left and adding new reward to the end of the history list
            self.rew_hist[:, :-1] = self.rew_hist[:, 1:] * 1
            self.rew_hist[:, -1] = self.rew_buf[:] * 1

        return

    def _init_tensor_history(self, env_ids):
        self._rigid_body_pos_hist[env_ids] = (
            self._rigid_body_pos[env_ids]
            .unsqueeze(1)
            .repeat(1, self.past_track_steps, 1, 1)
        )
        self._rigid_body_rot_hist[env_ids] = (
            self._rigid_body_rot[env_ids]
            .unsqueeze(1)
            .repeat(1, self.past_track_steps, 1, 1)
        )
        self._rigid_body_vel_hist[env_ids] = (
            self._rigid_body_vel[env_ids]
            .unsqueeze(1)
            .repeat(1, self.past_track_steps, 1, 1)
        )
        self._rigid_body_ang_vel_hist[env_ids] = (
            self._rigid_body_ang_vel[env_ids]
            .unsqueeze(1)
            .repeat(1, self.past_track_steps, 1, 1)
        )

    def _update_tensor_history(self):
        self._rigid_body_pos_hist = torch.cat(
            [self._rigid_body_pos_hist[:, 1:], self._rigid_body_pos.unsqueeze(1)], dim=1
        )
        self._rigid_body_rot_hist = torch.cat(
            [self._rigid_body_rot_hist[:, 1:], self._rigid_body_rot.unsqueeze(1)], dim=1
        )
        self._rigid_body_vel_hist = torch.cat(
            [self._rigid_body_vel_hist[:, 1:], self._rigid_body_vel.unsqueeze(1)], dim=1
        )
        self._rigid_body_ang_vel_hist = torch.cat(
            [
                self._rigid_body_ang_vel_hist[:, 1:],
                self._rigid_body_ang_vel.unsqueeze(1),
            ],
            dim=1,
        )

    def post_physics_step(self):
        # This is after stepping, so progress buffer got + 1. Compute reset/reward do not need to forward 1 timestep since they are for "this" frame now.
        # if not self.paused:
        #     self.progress_buf += 1

        # if self.self_obs_v == 2:
        #     self._update_tensor_history()
        self.reset_done()
        self._refresh_sim_tensors()
        # self._compute_reward(self.actions)  # ZL swapped order of reward & objecation computes. should be fine.
        self.rew_buf = self.rew_hist.mean(-1)
        self._compute_observations(
            self._rigid_body_pos.reshape(
                self.num_envs * num_agents, self.num_bodies, 3
            ),
            self._rigid_body_rot.reshape(
                self.num_envs * num_agents,
                self.num_bodies,
                self._rigid_body_rot.shape[-1],
            ),
            self._rigid_body_vel.reshape(
                self.num_envs * num_agents,
                self.num_bodies,
                self._rigid_body_vel.shape[-1],
            ),
            self._rigid_body_ang_vel.reshape(
                self.num_envs * num_agents,
                self.num_bodies,
                self._rigid_body_ang_vel.shape[-1],
            ),
        )  # observation for the next step.
        self._compute_reward(
            self.input_lats
        )  # ZL swapped order of reward & objecation computes. should be fine.
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf
        self.extras["reward_raw"] = self.reward_raw.detach()

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        # Debugging
        # if flags.debug:
        #     body_vel = self._rigid_body_vel.clone()
        #     speeds = body_vel.norm(dim = -1).mean(dim = -1)
        #     sorted_speed, sorted_idx = speeds.sort()
        #     print(sorted_speed.numpy()[::-1][:20], sorted_idx.numpy()[::-1][:20].tolist())
        #     # import ipdb; ipdb.set_trace()

        return

    def render(self, sync_frame_time=False, mode="human"):
        if self.viewer:
            self._update_camera()

        return super().render(mode)

    def _build_key_body_ids_tensor(self, key_body_names):
        if self.smpl_humanoid:
            body_ids = [self._body_names.index(name) for name in key_body_names]
            body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)

        else:
            env_ptr = self.envs[0]
            actor_handle = self.humanoid_handles[0]
            body_ids = []

            for body_name in key_body_names:
                body_id = self.gym.find_actor_rigid_body_handle(
                    env_ptr, actor_handle, body_name
                )
                assert body_id != -1
                body_ids.append(body_id)

            body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)

        return body_ids

    def _build_key_body_ids_orig_tensor(self, key_body_names):
        body_ids = [self._body_names_orig.index(name) for name in key_body_names]
        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name
            )
            assert body_id != -1
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        # TODO: do the repeat where they are defined.
        pd_tar = (
            self._pd_action_offset.repeat(num_agents)
            + self._pd_action_scale.repeat(num_agents) * action
        )
        return pd_tar


    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = (
            self._humanoid_root_states[0, main_agent, 0:3].cpu().numpy()
        )

        cam_pos = gymapi.Vec3(
            self._cam_prev_char_pos[0] + 5.0, self._cam_prev_char_pos[1] - 3.0, 3.0
        )
        cam_target = gymapi.Vec3(
            self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0
        )
        self.new_cam_pos_vis = (
            2.6125659942626953, -5.503269195556641, 1.7383817434310913
        )
        self.new_cam_pos_vis2 = (
            self._cam_prev_char_pos[0] + 2.0,
            self._cam_prev_char_pos[1] - 5.0,
            5.0,
        )
        # self.new_cam_pos_vis2 =  self.new_cam_pos_vis * 1
        #print(self.new_cam_pos_vis)
        self.new_cam_target_vis = (
            self._cam_prev_char_pos[0],
            self._cam_prev_char_pos[1],
            5.0,
        )
        self.new_cam_target_vis2 = self.new_cam_target_vis * 1
        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        #self._update_camera()
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._box_pos.reshape(self.num_envs,3)[0].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)

        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(
            char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2]
        )
        # print(new_cam_pos)
        # print(new_cam_target)
        self.new_cam_pos_vis = (
            2.6125659942626953, -5.503269195556641, 1.7383817434310913
        )
        self.new_cam_pos_vis2 = (16.250212, -7.945037, 6.500098)
        self.new_cam_target_vis2 =  (2.500003, 0.305096, 1.000000)
        #print(self.new_cam_pos_vis)
        self.new_cam_target_vis = (char_root_pos[0], char_root_pos[1], 1.0)
        # self.gym.set_camera_location(self.recorder_camera_handle, self.envs[0], new_cam_pos, new_cam_target)
        

        if self.viewer:
            self.gym.viewer_camera_look_at(
                self.viewer, None, new_cam_pos, new_cam_target
            )

        self._cam_prev_char_pos[:] = char_root_pos
        return
    
    def plot_scaler(self, title, scaler, step_count ):
        if(self.num_envs > 1): #don't plot scaler during evaluation
            self.writer.add_scalar(
                    title,
                    scaler,
                    step_count,
                )
    def update_plot(self, plot_index, x , y, title):
        # Plot red circles for current points
        i =plot_index// self.axs.shape[1]
        j =plot_index% self.axs.shape[1]
        #print(f"{i}, {j}")
        self.axs[i,j].plot(x, y, 'ro')  # Red circle for Data 1

        # Plot line between previous and current points
        if x > 1:
            prev_x = self.axs[i,j].get_lines()[-2].get_xdata()[-1]
            prev_y = self.axs[i,j].get_lines()[-2].get_ydata()[-1]
            self.axs[i,j].plot([prev_x, x], [prev_y, y], 'r-')  # Red line for Data 1
        
        # Set limits for better visualization (optional)
        # axs[0].set_xlim(0, 10)
        # axs[1].set_xlim(0, 10)
        
        # Add legend and pause
        self.axs[i,j].set_title(title)
        #self.axs[i,j].legend(['Data 1'])
        self.axs[i,j].relim()
        self.axs[i,j].autoscale_view()
        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events()

        #plt.draw() 
        # plt.savefig("mygraph.png")
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        # buf.seek(0)
        # image = PIL.Image.open(buf)
        # image = ToTensor()(image).unsqueeze(0)
        # self.fig.canvas.draw()
        # image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
        # image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        # plt.close(self.fig)
        

        # plt.show(block=True)
        # plt.close()
        #plt.pause(0.5)

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return

    def normalize_vector(self, v):
        norm = torch.norm(v)
        if norm == 0:
            return v
        return v / norm

    def rotate_vector_with_quaternion(self, vector, quaternion):
        # Ensure the quaternion is normalized
        quaternion = self.normalize_vector(quaternion)

        # Convert the 3D vector to a quaternion
        vec_quaternion = torch.cat([torch.tensor([0.0], device=self.device), vector])

        # Perform quaternion multiplication: qvq*
        rotated_vector_quaternion = self.quaternion_raw_multiply(
            self.quaternion_raw_multiply(quaternion, vec_quaternion),
            self.quaternion_conjugate(quaternion),
        )

        # Extract the rotated vector from the result quaternion
        rotated_vector = rotated_vector_quaternion[1:]

        return rotated_vector

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.tensor([w, x, y, z], device=self.device)

    def quaternion_raw_multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

        aw, ax, ay, az = torch.unbind(a, -1)
        bw, bx, by, bz = torch.unbind(b, -1)
        ow = aw * bw - ax * bx - ay * by - az * bz
        ox = aw * bx + ax * bw + ay * bz - az * by
        oy = aw * by - ax * bz + ay * bw + az * bx
        oz = aw * bz + ax * by - ay * bx + az * bw
        return torch.stack((ow, ox, oy, oz), -1)

    def quaternion_conjugate(self, q):
        return torch.tensor([q[0], -q[1], -q[2], -q[3]])


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def dof_to_obs_smpl(pose):
    # type: (Tensor) -> Tensor
    joint_obs_size = 6
    B, jts = pose.shape
    num_joints = int(jts / 3)

    joint_dof_obs = torch_utils.quat_to_tan_norm(
        torch_utils.exp_map_to_quat(pose.reshape(-1, 3))
    ).reshape(B, -1)
    assert (num_joints * joint_obs_size) == joint_dof_obs.shape[1]

    return joint_dof_obs


@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets):
    # ZL this can be sped up for SMPL
    # type: (Tensor, int, List[int]) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset : (dof_offset + dof_size)]

        # assume this is a spherical joint
        if dof_size == 3:
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif dof_size == 1:
            axis = torch.tensor(
                [0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device
            )
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            joint_pose_q = None
            assert False, "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size) : ((j + 1) * joint_obs_size)] = joint_dof_obs

    assert (num_joints * joint_obs_size) == dof_obs_size

    return dof_obs


@torch.jit.script
def compute_humanoid_observations(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    key_body_pos,
    local_root_obs,
    root_height_obs,
    dof_obs_size,
    dof_offsets,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    local_root_vel = torch_utils.my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    obs = torch.cat(
        (
            root_h_obs,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_humanoid_observations_max(
    body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs
):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    # TODO: this is only accounting for envs and not all agents, should be changed for multi agents
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = torch_utils.my_quat_rotate(
        flat_heading_rot, flat_local_body_pos
    )
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )  # global body rotation
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    if local_root_obs:
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(
        body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2]
    )
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]
    )

    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
    )
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(
        flat_heading_rot, flat_body_ang_vel
    )
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )

    obs = torch.cat(
        (
            root_h_obs,
            local_body_pos,
            local_body_rot_obs,
            local_body_vel,
            local_body_ang_vel,
        ),
        dim=-1,
    )
    return obs


# @torch.jit.script
def compute_humanoid_reward(
        self,
        _rigid_body_pos,
        red_rb_pos,
        blue_rb_pos,
        red_rb_root_xyz,
        prev_red_rb_root_xyz,
        blue_rb_root_xyz,
        prev_blue_rb_root_xyz,
        box_pos,
        hand_idx,
        head_idx

):
    # type: (Object, Tensor,Tensor,Tensor,Tensor,Tensor,Tensor, Tensor,Tensor, int, int) -> Tensor
    # # root_velocity = (red_rb_root_xyz - prev_red_rb_root_xyz) / 0.01666666666
    # # target_x_velocity = torch.tensor([[1.0, 0.0, 0.0]], device=red_rb_root_xyz.device)
    # # velocity_norm = torch.linalg.norm(root_velocity - target_x_velocity, dim=-1)
    # # velocity_reward = torch.clip((60 - velocity_norm) / 60, 0, 1)
    
    box_distance_mse_loss = torch.nn.functional.mse_loss(
        box_pos[..., :2], _rigid_body_pos[..., :2], reduction="none"
    )

    box_distance_loss_result = torch.mean(box_distance_mse_loss, dim=1)
    k1 = 1
    box_distance_reward = 1 - torch.exp(-1 * (10**k1) * box_distance_loss_result)

    # self.plot_scaler(
    #             "custom/box_distance_loss_result",
    #             torch.mean(box_distance_loss_result),
    #             self.step_count,
    #         )

    # self.plot_scaler(
    #             "reward/box_distance_reward",
    #             torch.mean(box_distance_reward),
    #             self.step_count,
    #         )
    box_distance_to_blue = (box_pos[..., :2].reshape(-1, 1, 2).repeat(1, 23, 1)- blue_rb_pos.reshape(-1, 24, 3)[:, 1:, :2])
    box_distance_to_blue_norm = torch.norm(box_distance_to_blue, dim=-1)
    max = torch.max(box_distance_to_blue_norm, dim=-1).values
    min = torch.min(box_distance_to_blue_norm, dim=-1).values
    min_idx = torch.min(box_distance_to_blue_norm, dim=-1).indices
    diff = (max-min).reshape(max.shape[0], 1)
    distance_weight_mat = (max[:, None] - box_distance_to_blue_norm ) / diff

    k1 = -1.3
    box_distance_delta_xyz = (
        box_pos[..., :2].reshape(-1, 1, 2).repeat(1, 23, 1)
        - red_rb_pos.reshape(-1, 24, 3)[:, 1:, :2]
    )
    box_distance_norm = torch.norm(box_distance_delta_xyz, dim=-1) 
    box_distance_norm_weighted = box_distance_norm * distance_weight_mat
    box_distance_sum = torch.sum(box_distance_norm_weighted, dim=-1) # the bigger the sum, the better the reward
    box_distance_sum_reward = 1 - torch.exp(-1 * (10**k1) * (box_distance_sum**2))
    
    self.plot_scaler("custom/box_distance_sum",torch.mean(box_distance_sum),self.step_count,)
    self.plot_scaler("reward/box_distance_sum_reward",torch.mean(box_distance_sum_reward),self.step_count,)

    
    k1 = 0.75
    box_distance_delta_xyz = (
        box_pos[..., :2].reshape(-1, 1, 2).repeat(1, 23, 1)
        - red_rb_pos.reshape(-1, 24, 3)[:, 1:, :2]
    )
    box_distance_norm = torch.norm(box_distance_delta_xyz, dim=-1)
    box_distance_min = torch.min(box_distance_norm, dim=-1).values
    box_distance_min_reward = 1 - torch.exp(-1 * (10**k1) * (box_distance_min**2))

    self.plot_scaler("custom/box_distance_min",torch.mean(box_distance_min),self.step_count,)
    self.plot_scaler("reward/box_distance_min_reward",torch.mean(box_distance_min_reward),self.step_count,)
    
    red_rb_pos_inv = red_rb_pos.reshape(-1, 24, 3) * 1
    red_rb_pos_inv -= red_rb_pos_inv[:, [0]]
    blue_rb_pos_inv = blue_rb_pos.reshape(-1, 24, 3) * 1
    blue_rb_pos_inv -= blue_rb_pos_inv[:,[0]]

    k5 = 0.7
    imitation_inv = (red_rb_pos_inv.reshape(-1, 24, 3)- blue_rb_pos_inv.reshape(-1, 24, 3))
    imitation_inv_mean_norm = torch.mean(torch.norm(imitation_inv, dim=-1), dim=-1)
    imitation_inv_reward = 1e0 * torch.exp(-1 * (10**k5) * (imitation_inv_mean_norm**2))


    self.plot_scaler("custom/imitation_inv_mean_norm",torch.mean(imitation_inv_mean_norm),self.step_count,)
    self.plot_scaler("reward/imitation_inv_reward", torch.mean(imitation_inv_reward), self.step_count,)

    delta_root_xyz = (red_rb_pos.reshape(-1, 24, 3)[:,[0]]- blue_rb_pos.reshape(-1, 24, 3)[:,[0]])
    delta_root_xyz_mean_norm = torch.mean(torch.norm(delta_root_xyz, dim=-1), dim=-1)
    delta_root_xyz_mean_norm_reward = 1e0 * torch.exp(-(delta_root_xyz_mean_norm**2))

    self.plot_scaler("custom/delta_root_xyz_mean_norm", torch.mean(delta_root_xyz_mean_norm), self.step_count,)
    self.plot_scaler("reward/delta_root_xyz_mean_norm_reward",torch.mean(delta_root_xyz_mean_norm_reward), self.step_count,)
    self.plot_scaler("custom/box_heigh", torch.mean(-10 *box_pos[...,2]), self.step_count,)

    k4=1.3
    red_velocity_mag = torch.norm((red_rb_root_xyz - prev_red_rb_root_xyz), dim = -1)
    blue_velocity_mag = torch.norm((blue_rb_root_xyz - prev_blue_rb_root_xyz), dim = -1)
    velocity_mag_distance = red_velocity_mag - blue_velocity_mag
    velocity_reward = 1e0 * torch.exp(-1 * (10**k4) * (velocity_mag_distance**2))

    if(self.calc_root_dir):
        k5 = 2.8
        yaw_delta = self.current_red_yaw_angles[...,0] - self.prev_red_yaw_angles[...,0]
        yaw_reward = 1e0 * torch.exp(-1 * (10**k5) * (yaw_delta**2))
    # print(yaw_reward)
    
    self.plot_scaler("custom/velocity_mag_distance", torch.mean(velocity_mag_distance), self.step_count,)

    self.plot_scaler("reward/velocity_reward", torch.mean(velocity_reward), self.step_count,)
    
    self.plot_scaler( "reward/root_and_box_reward",torch.mean(delta_root_xyz_mean_norm_reward).item() + torch.mean(box_distance_min_reward).item(),self.step_count,)
    
    if(self.num_envs == 1):
        self.easy_plot.add_point(self.step_count, torch.mean(box_distance_sum).item(), 'box_distance_sum')
        self.easy_plot.add_point(self.step_count, torch.mean(box_distance_sum_reward).item(), 'box_distance_sum_reward')
        self.easy_plot.add_point(self.step_count, torch.mean(box_distance_min).item(), 'box_distance_min')
        self.easy_plot.add_point(self.step_count, torch.mean(box_distance_min_reward).item(), 'box_distance_min_reward')
        self.easy_plot.add_point(self.step_count, torch.mean(imitation_inv_mean_norm).item(), 'imitation_inv_mean_norm')
        self.easy_plot.add_point(self.step_count, torch.mean(imitation_inv_reward).item(), 'imitation_inv_reward')

        self.easy_plot.add_point(self.step_count, torch.mean(delta_root_xyz_mean_norm).item(), 'delta_root_xyz_mean_norm')
        self.easy_plot.add_point(self.step_count, torch.mean(delta_root_xyz_mean_norm_reward).item(), 'delta_root_xyz_mean_norm_reward')
        self.easy_plot.add_point(self.step_count, torch.mean(-10 *box_pos[...,2]).item(), 'box_heigh')
        self.easy_plot.add_point(self.step_count, torch.mean(velocity_mag_distance).item(), 'velocity_mag_distance')
        self.easy_plot.add_point( self.step_count, torch.mean(velocity_reward).item(), 'velocity_reward')
        if(self.calc_root_dir):
            self.easy_plot.add_point( self.step_count, torch.mean(yaw_delta).item(), 'yaw_delta')
            self.easy_plot.add_point( self.step_count, torch.mean(yaw_reward).item(), 'yaw_reward')
        self.easy_plot.add_point( self.step_count, torch.mean(delta_root_xyz_mean_norm_reward).item() + torch.mean(box_distance_min_reward).item(), 'root_and_box_reward')
        if(self.step_count % 100 == 0):
            print(self.step_count)
        if(self.step_count == self.max_episode_length - 1):
            print(self.step_count)
            self.easy_plot.plot(self.writer, self.epoch_count)

    coeff = [ 0.1       ,  0.16681005,  0.27825594,  0.46415888,  0.77426368,1.29154967,  2.15443469,  3.59381366,  5.9948425 , 10.        ]
    reward = (
        1 * imitation_inv_reward
        + 3 * delta_root_xyz_mean_norm_reward
        +  2 * box_distance_min_reward
        + 2 * velocity_reward
        #+ yaw_reward
    )
    return reward



@torch.jit.script
def compute_humanoid_reset(
    step_count,
    hand_idx,
    head_idx,
    reset_buf,
    progress_buf,
    contact_buf,
    contact_body_ids,
    rigid_body_pos,
    termination_heights,
    blue_rb_xyz,
    red_rb_xyz,
    red_root_sixd,
    red_rb_root_xyz,
    prev_red_rb_root_xyz,
    _box_pos,
    current_motion_times,
    _motion_lengths,
    max_episode_length,
    enable_early_termination,
):
    # type: (int, int,int, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor,Tensor, Tensor, float, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    # enable_early_termination
    if enable_early_termination:

        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        ############################## Debug ##############################
        # mujoco_joint_names = np.array(['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']);  print( mujoco_joint_names[masked_contact_buf[0, :, 0].nonzero().cpu().numpy()])
        ############################## Debug ##############################

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # terminate when contact happens
        # has_fallen = torch.logical_or(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps

        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)


        delta = blue_rb_xyz.reshape(-1, 24, 3) - red_rb_xyz.reshape(-1, 24, 3)
        delta_mean_norm = torch.mean(torch.norm(delta, dim=-1), dim=-1)
        # target_x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=blue_rb_xyz.device)
        # cossim_x = torch.cosine_similarity(
        #     red_root_sixd[..., :3], target_x_axis, dim=-1
        # )

        # print(delta_mean_norm)
        #velocity = red_rb_root_xyz - prev_red_rb_root_xyz
        # target_x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=blue_rb_xyz.device)
        # velocity_cossim_x = torch.cosine_similarity(velocity, target_x_axis, dim=-1)
        

        # root_delta_xyz = red_rb_root_xyz - prev_red_rb_root_xyz
        # red_root_distance = torch.mean(torch.norm(root_delta_xyz, dim=-1), dim=-1)
        #print(delta_mean_norm)
        #hand_xyz = red_rb_xyz.reshape(-1, 24, 3)[:, hand_idx]
        #target = hand_xyz * 1
        #target[..., 2] = 2
        #target_hand_xyz = hand_xyz * 1
        #delta_hand_xyz = hand_xyz - target
        #delta_hand_xyz = hand_xyz[..., [2]] - 2
        #delta_hand_xyz_mean_norm = torch.norm(delta_hand_xyz, dim=-1)
        #print(delta_hand_xyz_mean_norm[0])
        #threshold = max((2.0 - 0.2 * (step_count/(800))),  0.2)
        #print((threshold))
        #terminated |= delta_hand_xyz_mean_norm > threshold
        
        delta_root_xyz = (red_rb_xyz.reshape(-1, 24, 3)[:,[0]]- blue_rb_xyz.reshape(-1, 24, 3)[:,[0]])
        delta_root_xyz_mean_norm = torch.mean(torch.norm(delta_root_xyz, dim=-1), dim=-1)

        terminated |= delta_mean_norm > 0.8
        #terminated = test > 1
        animation_ended = current_motion_times > _motion_lengths
        #terminated |= animation_ended.squeeze(-1)

        
        box_fallen = _box_pos[...,2] > 0.1

        #terminated |= box_fallen

        #test = torch.zeros_like(terminated)

        #terminated = test

        terminated *= progress_buf > 2

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )
    return reset, terminated


# @torch.jit.script
# def compute_humanoid_reset(
#     reset_buf,
#     progress_buf,
#     contact_buf,
#     contact_body_ids,
#     rigid_body_pos,
#     max_episode_length,
#     enable_early_termination,
#     termination_heights,
# ):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
#     # ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
#     terminated = torch.zeros_like(reset_buf)
#     # enable_early_termination
#     if enable_early_termination:
#
#         masked_contact_buf = contact_buf.clone()
#         masked_contact_buf[:, contact_body_ids, :] = 0
#         fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
#         # print(torch.any(torch.abs(masked_contact_buf) > 500, dim=-1))
#         fall_contact = torch.any(fall_contact, dim=-1)
#         # if fall_contact.any():
#         # print(masked_contact_buf[0, :, 0].nonzero())
#         #     import ipdb
#         #     ipdb.set_trace()
#
#         body_height = rigid_body_pos[..., 2]
#         fall_height = body_height < termination_heights
#         fall_height[:, contact_body_ids] = False
#         fall_height = torch.any(fall_height, dim=-1)
#
#         ############################## Debug ##############################
#         # mujoco_joint_names = np.array(['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']);  print( mujoco_joint_names[masked_contact_buf[0, :, 0].nonzero().cpu().numpy()])
#         ############################## Debug ##############################
#
#         has_fallen = torch.logical_and(fall_contact, fall_height)
#
#         # terminate when contact happens
#         # has_fallen = torch.logical_or(fall_contact, fall_height)
#
#         # first timestep can sometimes still have nonzero contact forces
#         # so only check after first couple of steps
#         has_fallen *= progress_buf > 1
#         terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
#
#     reset = torch.where(
#         progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
#     )
#     # import ipdb
#     # ipdb.set_trace()
#
#     return reset, terminated


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def remove_base_rot(quat):
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat))  # SMPL
    shape = quat.shape[0]
    return quat_mul(quat, base_rot.repeat(shape, 1))


@torch.jit.script
def compute_humanoid_observations_smpl(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    key_body_pos,
    dof_obs_size,
    dof_offsets,
    smpl_params,
    local_root_obs,
    root_height_obs,
    upright,
    has_smpl_params,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, List[int], Tensor, bool, bool,bool, bool) -> Tensor
    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    local_root_vel = torch_utils.my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    obs_list = []
    if root_height_obs:
        obs_list.append(root_h_obs)
    obs_list += [
        root_rot_obs,
        local_root_vel,
        local_root_ang_vel,
        dof_obs,
        dof_vel,
        flat_local_key_pos,
    ]
    if has_smpl_params:
        obs_list.append(smpl_params)
    obs = torch.cat(obs_list, dim=-1)

    return obs


@torch.jit.script
def compute_humanoid_observations_smpl_max(
    body_pos,
    body_rot,
    body_vel,
    body_ang_vel,
    smpl_params,
    limb_weight_params,
    local_root_obs,
    root_height_obs,
    upright,
    has_smpl_params,
    has_limb_weight_params,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(
        heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],
        heading_rot_inv_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = torch_utils.my_quat_rotate(
        flat_heading_rot_inv, flat_local_body_pos
    )
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    if not (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(
            root_rot
        )  # If not local root obs, you override it.
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(
        body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2]
    )
    flat_local_body_vel = torch_utils.my_quat_rotate(
        flat_heading_rot_inv, flat_body_vel
    )
    local_body_vel = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]
    )

    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
    )
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(
        flat_heading_rot_inv, flat_body_ang_vel
    )
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )

    obs_list = []
    if root_height_obs:
        obs_list.append(root_h_obs)
    obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel]
    # obs_list.append(root_pos)

    if has_smpl_params:
        obs_list.append(smpl_params)

    if has_limb_weight_params:
        obs_list.append(limb_weight_params)

    obs = torch.cat(obs_list, dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_observations_smpl_max_v2(
    body_pos,
    body_rot,
    body_vel,
    body_ang_vel,
    smpl_params,
    limb_weight_params,
    local_root_obs,
    root_height_obs,
    upright,
    has_smpl_params,
    has_limb_weight_params,
    time_steps,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool, int) -> Tensor
    root_pos = body_pos[:, -1, 0, :]
    root_rot = body_rot[:, -1, 0, :]
    B, T, J, C = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    root_h_obs = root_pos[:, 2:3]
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    # heading_rot_inv_expand = heading_inv_rot.unsqueeze(-2)
    # heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    # flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2])

    heading_inv_rot_expand = (
        heading_inv_rot.unsqueeze(-2)
        .repeat((1, J, 1))
        .repeat_interleave(time_steps, 0)
        .view(-1, 4)
    )
    heading_rot_expand = (
        heading_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    )

    root_pos_expand = root_pos.unsqueeze(-2).unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = torch_utils.my_quat_rotate(
        heading_inv_rot_expand, local_body_pos.view(-1, 3)
    )
    local_body_pos = flat_local_body_pos.reshape(B, time_steps, J * C)
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_local_body_rot = quat_mul(heading_inv_rot_expand, body_rot.view(-1, 4))
    local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot).view(
        B, time_steps, J * 6
    )

    if not (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(
            body_rot[:, :, 0].view(-1, 4)
        )  # If not local root obs, you override it.
        local_body_rot_obs[..., 0:6] = root_rot_obs

    local_body_vel = torch_utils.my_quat_rotate(
        heading_inv_rot_expand, body_vel.view(-1, 3)
    ).view(B, time_steps, J * 3)

    local_body_ang_vel = torch_utils.my_quat_rotate(
        heading_inv_rot_expand, body_ang_vel.view(-1, 3)
    ).view(B, time_steps, J * 3)

    ##################### Compute_history #####################
    body_obs = torch.cat(
        [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel], dim=-1
    )

    obs_list = []
    if root_height_obs:
        body_obs = torch.cat([body_pos[:, :, 0, 2:3], body_obs], dim=-1)

    obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel]

    if has_smpl_params:
        raise NotImplementedError

    if has_limb_weight_params:
        raise NotImplementedError

    obs = body_obs.view(B, -1)
    return obs


@torch.jit.script
def compute_imitation_observations_v7(
    root_pos,
    root_rot,
    body_pos,
    body_vel,
    ref_body_pos,
    ref_body_vel,
    time_steps,
    upright,
):
    # type: (Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, int, bool) -> Tensor
    # No rotation information. Leave IK for RL.
    # Future tracks in this obs will not contain future diffs.

    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_inv_rot_expand = (
        heading_inv_rot.unsqueeze(-2)
        .repeat((1, body_pos.shape[1], 1))
        .repeat_interleave(time_steps, 0)
    )

    ##### Body position differences
    diff_global_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - body_pos.reshape(
        B, 1, J, 3
    )
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), diff_global_body_pos.reshape(-1, 3)
    )

    ##### Linear Velocity differences
    diff_global_vel = ref_body_vel.reshape(B, time_steps, J, 3) - body_vel.reshape(B, 1, J, 3)
    diff_local_vel = torch_utils.my_quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), diff_global_vel.reshape(-1, 3)
    )

    ##### body pos + Dof_pos
    local_ref_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - root_pos.reshape(
        B, 1, 1, 3
    )  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), local_ref_body_pos.reshape(-1, 3)
    )

    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.reshape(B, time_steps, -1))  # 1 * 10 * 3 * 3
    obs.append(diff_local_vel.reshape(B, time_steps, -1))  # 3 * 3
    obs.append(local_ref_body_pos.reshape(B, time_steps, -1))  # 2

    obs = torch.cat(obs, dim=-1).reshape(B, -1)
    return obs
