from time import time
from warnings import WarningMessage
import numpy as np
import pinocchio as pin
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from rich.progress import Progress

from humanoidverse.envs.env_utils.general import class_to_dict
from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_wbc_new import LeggedRobotDecoupledLocomotionStanceWBC
# from humanoidverse.envs.env_utils.command_generator import CommandGenerator
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_inverse,
)
from humanoidverse.envs.env_utils.visualization import Point

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

from humanoidverse.envs.locomotion.locomotion import LeggedRobotLocomotion

from loguru import logger

from scipy.stats import vonmises

from humanoidverse.utils.arm_ik import arm_ik

DEBUG = False
class LeggedRobotDecoupledLocomotionStanceHeightWBC(LeggedRobotDecoupledLocomotionStanceWBC):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.command_height_scale = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 6), dtype=torch.float32, device=self.device
        )

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        # Sample the desired base height
        self.commands[env_ids, 5] = self.config.rewards.desired_base_height
        self.commands[env_ids, 5] += torch_rand_float(self.command_ranges["base_height"][0], 
                                                      self.command_ranges["base_height"][1], 
                                                      (len(env_ids), 1), device=self.device).squeeze(1) * (1.0 - self.commands[env_ids, 4]) # only apply the base height if standing
    
    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        
        self.commands = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self.commands[:, 5] = self.config.rewards.desired_base_height
        # Apply full upper body action scale
        self.action_scale_upper_body = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # TODO: haotian: adding command configuration
        if command is not None:
            self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands

        # self.config.obs.noise_scales = {
        #     key: value * 0.0 for key, value in self.config.obs.noise_scales.items()
        # }
        # print("noise scales: ", self.config.obs.noise_scales)
    
    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """ Resets the environments with the given ids and optionally to the target states
        """
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # if self.config.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
        #     self.update_command_curriculum(env_ids)
        if self.config.rewards.upper_body_motion_scale_curriculum:
            self._update_upper_body_motion_scale_curriculum(env_ids)
        else:
            self.action_scale_upper_body[env_ids] = 1.0
        if self.config.rewards.get("command_height_scale_curriculum", False):
            self._update_command_height_curriculum(env_ids)
        else:
            self.command_height_scale[env_ids] = 1.0
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)        # if target_states is not None, reset to target states
        self._reset_robot_states_callback(env_ids, target_states)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["time_outs"] = self.time_out_buf
        # self._refresh_sim_tensors()

    # ########################### Curriculum #############################
    def _update_command_height_curriculum(self, env_ids):
        """
        Update the command height scale based on the episode length for each environment.
        Returns:
            None
        """
        env_ids_scale_up_mask = self.episode_length_buf[env_ids] > self.config.rewards.command_height_scale_up_threshold
        env_ids_scale_up = env_ids[torch.where(env_ids_scale_up_mask)[0]]
        env_ids_scale_down_mask = self.episode_length_buf[env_ids] < self.config.rewards.command_height_scale_down_threshold
        env_ids_scale_down = env_ids[torch.where(env_ids_scale_down_mask)[0]]
        self.command_height_scale[env_ids_scale_up] += self.config.rewards.command_height_scale_up
        self.command_height_scale[env_ids_scale_down] -= self.config.rewards.command_height_scale_down
        # Clip the scale
        self.command_height_scale[env_ids] = torch.clip(self.command_height_scale[env_ids], 
                                                        self.config.rewards.command_height_scale_min, 
                                                        self.config.rewards.command_height_scale_max)

    ########################### FEET REWARDS ###########################
    def _reward_penalty_hip_pos(self):
        # Penalize the hip joints (only roll and yaw)
        hips_roll_yaw_indices = self.hips_dof_id[1:3] + self.hips_dof_id[4:6]
        hip_pos = self.simulator.dof_pos[:, hips_roll_yaw_indices]
        penalty_hip_pos = torch.sum(torch.square(hip_pos), dim=1)
        return penalty_hip_pos
        return penalty_hip_pos * (self.commands[:, 4] + (1 - self.commands[:, 4]) * self.commands[:, 5])

    def _reward_penalty_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2]) * self.commands[:, 4] # only apply the base linear z-axis velocity penalty if locomoting
    
    def _reward_penalty_torso_orientation(self):
        # Penalize non flat torso orientation
        torso_quat = self.simulator._rigid_body_rot[:, self.torso_index]
        projected_gravity_torso = quat_rotate_inverse(torso_quat, self.gravity_vec)
        torso_quat_target = quat_from_euler_xyz(self.ref_upper_dof_pos[:, 1], # roll
                                                self.ref_upper_dof_pos[:, 2], # pitch
                                                self.ref_upper_dof_pos[:, 0]) # yaw
        projected_gravity_torso_target = quat_rotate_inverse(torso_quat_target, self.gravity_vec)
        projected_gravity_torso_error = projected_gravity_torso - projected_gravity_torso_target
        return torch.sum(torch.square(projected_gravity_torso_error[:, :2]), dim=1) * (1 - self.commands[:, 4]) + \
               torch.sum(torch.square(projected_gravity_torso[:, :2]), dim=1) * self.commands[:, 4] * self.apply_waist_roll_pitch_only_when_stance
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.simulator.robot_root_states[:, 2]
        # return torch.square(base_height - self.config.rewards.desired_base_height)*self.commands[:, 4] # only apply the base height penalty if locomoting
        penalty_base_height = torch.square(base_height - self.commands[:, 5]) # only apply the base height penalty if standing
        stance_env_idx = torch.where(self.commands[:, 4] < 1)[0]
        penalty_base_height[stance_env_idx] *= 2.5 # double the penalty if standing
        return penalty_base_height
    
    def _reward_tracking_base_height(self):
        # Tracking of base height commands (z axe)
        base_height_error = torch.abs(self.commands[:, 5] - self.simulator.robot_root_states[:, 2])
        return torch.exp(-base_height_error/self.config.rewards.reward_tracking_sigma.base_height)*(1.0 - self.commands[:, 4]) # only apply the base height reward if standing
    
    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity x commands
        lin_vel_x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_x_error/self.config.rewards.reward_tracking_sigma.lin_vel)
    
    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity y commands
        lin_vel_y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_y_error/self.config.rewards.reward_tracking_sigma.lin_vel)
    
    ######################### Observations #########################
    def _get_obs_command_base_height(self):
        return self.commands[:, 5:6]