import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common import vecenv
from rl_games.common import common_losses

import torch
from torch import optim
from torch import nn

from tensorboardX import SummaryWriter

def MSE_distance(first, second):
    loss = torch.nn.functional.mse_loss(
                first[..., :2], second[..., :2], reduction="none"
            )

    return torch.mean(loss, dim=1)

class MyA2c_continuous(a2c_continuous.A2CAgent):
    def __init__(self, base_name, config):
        a2c_continuous.A2CAgent.__init__(self, base_name, config)

        self.model.a2c_network.sigma.requires_grad = False

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        # print()
        # print('------------------------------------------------------')
        # print()

        # for key, value in input_dict.items():

        #     print(f"{key}_min: {value.min()}")
        #     print(f"{key}_max: {value.max()}")
        #     print(f"{key}_Nan: {torch.isnan(value).any().item()}")

        #     print(f"{key}: {value}")
        obs_batch = self._preproc_obs(obs_batch)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            a_loss = common_losses.actor_loss(
                old_action_log_probs_batch,
                action_log_probs,
                advantage,
                self.ppo,
                curr_e_clip,
            )

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(
                    value_preds_batch,
                    values,
                    curr_e_clip,
                    return_batch,
                    self.clip_value,
                )
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)

            b_loss = self.bound_loss(mu)
            losses, sum_mask = torch_ext.apply_masks(
                [
                    a_loss.unsqueeze(1),
                    c_loss,
                    entropy.unsqueeze(1),
                    b_loss.unsqueeze(1),
                ],
                rnn_masks,
            )
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = (
                a_loss
                + 0.5 * c_loss * self.critic_coef
                - entropy * self.entropy_coef
                + b_loss * self.bounds_loss_coef
            )

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        # TODO: Refactor this ugliest code of they year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(
                mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl
            )
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        self.train_result = (
            a_loss,
            c_loss,
            entropy,
            kl_dist,
            self.last_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss,
        )

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        while True:
            epoch_num = self.update_epoch()
            (
                step_time,
                play_time,
                update_time,
                sum_time,
                a_losses,
                c_losses,
                b_losses,
                entropies,
                kls,
                last_lr,
                lr_mul,
            ) = self.train_epoch()
            self.model.a2c_network.sigma.data = torch.clip(
                self.model.a2c_network.sigma - 0.001, -5.0, 0.5
            )

            total_time += sum_time
            frame = self.frame
            self.writer.add_scalar(
                "custom/sigma_mean",
                torch.mean(self.model.a2c_network.sigma.data),
                frame,
            )

            # my_env = self.vec_env.env

            # green_rb_pos = my_env._rigid_body_pos.reshape(my_env.num_envs, -1)
            # red_rb_pos = my_env.modified_ref_body_pos.reshape(my_env.num_envs, -1)
            # blue_rb_pos = my_env.modified_rb_body_pos.reshape(my_env.num_envs, -1)
            # box_pos = my_env._box_pos.reshape(my_env.num_envs, -1)
            

            # self.writer.add_scalar(
            #     "custom/box_to_blue",
            #     MSE_distance(box_pos,blue_rb_pos).mean(),
            #     epoch_num,
            # )

            # self.writer.add_scalar(
            #     "custom/box_to_red",
            #     MSE_distance(box_pos,red_rb_pos).mean(),
            #     epoch_num,
            # )

            # self.writer.add_scalar(
            #     "custom/box_to_green",
            #     MSE_distance(box_pos,green_rb_pos).mean(),
            #     epoch_num,
            # )

            




            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            if self.multi_gpu:
                self.hvd.sync_stats(self)
            should_exit = False
            if self.rank == 0:
                # do we need scaled_time?
                scaled_time = sum_time  # self.num_agents * sum_time
                scaled_play_time = play_time  # self.num_agents * play_time
                curr_frames = self.curr_frames
                self.frame += curr_frames
                if self.print_stats:
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(
                        f"fps step: {fps_step:.1f} fps step and policy inference: {fps_step_inference:.1f}  fps total: {fps_total:.1f}"
                    )

                self.write_stats(
                    total_time,
                    epoch_num,
                    step_time,
                    play_time,
                    update_time,
                    a_losses,
                    c_losses,
                    entropies,
                    kls,
                    last_lr,
                    lr_mul,
                    frame,
                    scaled_time,
                    scaled_play_time,
                    curr_frames,
                )
                if len(b_losses) > 0:
                    self.writer.add_scalar(
                        "losses/bounds_loss",
                        torch_ext.mean_list(b_losses).item(),
                        frame,
                    )

                if self.has_soft_aug:
                    self.writer.add_scalar(
                        "losses/aug_loss", np.mean(aug_losses), frame
                    )

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()

                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = "rewards" if i == 0 else "rewards{0}".format(i)
                        self.writer.add_scalar(
                            rewards_name + "/step".format(i), mean_rewards[i], frame
                        )
                        self.writer.add_scalar(
                            rewards_name + "/iter".format(i), mean_rewards[i], epoch_num
                        )
                        self.writer.add_scalar(
                            rewards_name + "/time".format(i),
                            mean_rewards[i],
                            total_time,
                        )

                    self.writer.add_scalar("episode_lengths/step", mean_lengths, frame)
                    self.writer.add_scalar(
                        "episode_lengths/iter", mean_lengths, epoch_num
                    )
                    self.writer.add_scalar(
                        "episode_lengths/time", mean_lengths, total_time
                    )

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    # checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])
                    checkpoint_name = str(epoch_num)

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (
                            mean_rewards[0] <= self.last_mean_rewards
                        ):

                            self.save(os.path.join(self.nn_dir, checkpoint_name))

                    if (
                        mean_rewards[0] > self.last_mean_rewards
                        and epoch_num >= self.save_best_after
                    ):
                        print("saving next best rewards: ", mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config["name"]))
                        if self.last_mean_rewards > self.config["score_to_win"]:
                            print("Network won!")
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            should_exit = True

                if epoch_num > self.max_epochs:
                    self.save(
                        os.path.join(
                            self.nn_dir,
                            "last_"
                            + self.config["name"]
                            + "ep"
                            + str(epoch_num)
                            + "rew"
                            + str(mean_rewards),
                        )
                    )
                    print("MAX EPOCHS NUM!")
                    should_exit = True

                update_time = 0
            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit).float()
                self.hvd.broadcast_value(should_exit_t, "should_exit")
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num
