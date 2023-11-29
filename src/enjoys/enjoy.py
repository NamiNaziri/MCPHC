# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2022, NVIDIA Corporation
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
import argparse

import isaacgym
import isaacgymenvs
import torch


## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)


def main(args):
    if args.debug_yes:
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    envs = isaacgymenvs.make(
        seed=0,
        task="Ant",
        # task="Cartpole",
        # task="Pendulum",
        headless=False,
        num_envs=1,
        graphics_device_id=0,
        sim_device="cuda:0",
        rl_device="cuda:0",
        force_render=True,
    )
    # envs.is_vector_env = True
    envs.render()
    print("Observation space is", envs.observation_space)
    print("Action space is", envs.action_space)
    obs = envs.reset()
    for _ in range(20000):
        # obs, reward, done, info = envs.step(torch.tensor([1.], device="cuda:0"))
        obs, reward, done, info = envs.step(torch.zeros(1, *envs.action_space.shape, device="cuda:0"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_yes", action="store_true")
    args = parser.parse_args()
    main(args)
