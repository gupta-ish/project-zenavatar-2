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
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
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

from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand import LeggedRobotDecoupledLocomotionStance

from loguru import logger

from scipy.stats import vonmises

from humanoidverse.utils.arm_ik import arm_ik

DEBUG = False
class LeggedRobotDecoupledLocomotionStanceWaist(LeggedRobotDecoupledLocomotionStance):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)

    def _init_tracking_config(self):
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_link]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.upper_body_link]
            self.pelvis_id = self.simulator._body_list.index(self.config.robot.motion.pelvis_link)
        if "hips_link" in self.config.robot.motion:
            self.hips_dof_id = [self.simulator._body_list.index(link) - 1 for link in self.config.robot.motion.hips_link] # Yuanhang: -1 for the base link (pelvis)
        if "waist_link" in self.config.robot.motion:
            self.waist_dof_id = [self.simulator._body_list.index(link) - 1 for link in self.config.robot.motion.waist_link]
        if self.config.resample_motion_when_training:
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.dt)
        
    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 8), dtype=torch.float32, device=self.device
        )
        self.motion_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.command_ranges = self.config.locomotion_command_ranges
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.ref_upper_dof_pos = torch.zeros(self.num_envs, self.config.robot.upper_body_actions_dim, \
                                               dtype=torch.float32, device=self.device, requires_grad=False)
        self.episode_motion_length = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.tapping_in_place = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.fix_waist_yaw_range = self.config.fix_waist_yaw_range
        self.fix_waist_pitch_range = self.config.fix_waist_pitch_range
        self.fix_waist_roll_range = self.config.fix_waist_roll_range
        self.zero_fix_waist_roll = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.zero_fix_waist_pitch = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.zero_fix_waist_yaw = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.fixed_waist_yaw_pos = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.fixed_waist_pitch_pos = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.fixed_waist_roll_pos = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.apply_waist_yaw_only_when_stance = self.config.apply_waist_yaw_only_when_stance
        self.apply_waist_roll_only_when_stance = self.config.apply_waist_roll_only_when_stance
        self.apply_waist_pitch_only_when_stance = self.config.apply_waist_pitch_only_when_stance
        self.apply_waist_roll_pitch_only_when_stance = 1 if (self.config.apply_waist_roll_only_when_stance and self.config.apply_waist_pitch_only_when_stance) else 0

    def _pre_compute_observations_callback(self, debug=DEBUG):
        super()._pre_compute_observations_callback()
        # TODO [Yuanhang]: hardcode for Tairan's need :) should be removed
        if self.config.rewards.fix_upper_body:
            self.ref_upper_dof_pos *= 0.0
            self.ref_upper_dof_pos[:, 4] = 0.3
            self.ref_upper_dof_pos[:, 11] = -0.3
            self.ref_upper_dof_pos[:, 6] = 1.
            self.ref_upper_dof_pos[:, 13] = 1.
            return
        # Get the reference upper body joint positions
        offset = self.env_origins
        # print("env_ids_stance: ", env_ids_stance)
        self.motion_times = (self.episode_motion_length + 1) * self.dt + self.motion_start_times # next frames so +1
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=offset)
        
        # Update the upper body joint positions from motion library
        ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        self.ref_upper_dof_pos = ref_joint_pos[:, -self.config.robot.upper_body_actions_dim:] # [num_envs, upper_body_actions_dim]
        # Yuanhang: set waist's yaw, roll and pitch [12/21/2024]
        # if self.fix_waist_yaw:
        #     # self.ref_upper_dof_pos[:, 0] *= 0.0
        #     self.ref_upper_dof_pos[:, 0] = self.fixed_waist_yaw_pos
        # if self.fix_waist_roll:
        #     # self.ref_upper_dof_pos[:, 1] *= 0.0
        #     self.ref_upper_dof_pos[:, 1] = self.fixed_waist_roll_pos
        # if self.fix_waist_pitch:
        #     # self.ref_upper_dof_pos[:, 2] *= 0.0
        #     self.ref_upper_dof_pos[:, 2] = self.fixed_waist_pitch_pos 
        # if self.apply_waist_roll_only_when_stance:
        #     self.ref_upper_dof_pos[:, 1] *= (1 - self.commands[:, 4]) # only apply when stance
        # if self.apply_waist_pitch_only_when_stance:
        #     self.ref_upper_dof_pos[:, 2] *= (1 - self.commands[:, 4]) # only apply when stance
        # if self.apply_waist_yaw_only_when_stance:
        #     self.ref_upper_dof_pos[:, 0] *= (1 - self.commands[:, 4]) # only apply when stance
        # Yuanhang: only test for evaluation
        # if self.is_evaluating:
        #     self.ref_upper_dof_pos[:, :] *= 0.0
        # print("waist yaw: ", self.ref_upper_dof_pos[:, 0])
        # print("waist roll: ", self.ref_upper_dof_pos[:, 1])
        # print("waist pitch: ", self.ref_upper_dof_pos[:, 2])
        # Apply upper body action scale
        self.ref_upper_dof_pos *= self.action_scale_upper_body

    def _resample_commands(self, env_ids):
        if not self.config.robot.motion.reverse_motion: self._resample_motion_times(env_ids)
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # Sample the tapping or stand command with a probability
        self.commands[env_ids, 4] = (torch.rand(len(env_ids), device=self.device) > self.stand_prob).float()
        # Sample the tapping in place command with a probability
        self.tapping_in_place[env_ids, 0] = (torch.rand(len(env_ids), device=self.device) > self.tapping_in_place_prob).float()
        self.commands[env_ids, 0] *= (self.commands[env_ids, 4] * self.tapping_in_place[env_ids, 0])
        self.commands[env_ids, 1] *= (self.commands[env_ids, 4] * self.tapping_in_place[env_ids, 0])
        self.commands[env_ids, 2] *= (self.commands[env_ids, 4] * self.tapping_in_place[env_ids, 0])
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        
        # Sample the waist yaw, pitch and roll fixed dof pos
        self.zero_fix_waist_yaw[env_ids] = (torch.rand(len(env_ids), device=self.device) > self.config.zero_fix_waist_yaw_prob).float()
        self.zero_fix_waist_roll[env_ids] = (torch.rand(len(env_ids), device=self.device) > self.config.zero_fix_waist_roll_prob).float()
        self.zero_fix_waist_pitch[env_ids] = (torch.rand(len(env_ids), device=self.device) > self.config.zero_fix_waist_pitch_prob).float()
        if self.apply_waist_yaw_only_when_stance:
            self.zero_fix_waist_yaw[env_ids] *= (1 - self.commands[env_ids, 4]) # only apply when stance
        if self.apply_waist_roll_only_when_stance:
            self.zero_fix_waist_roll[env_ids] *= (1 - self.commands[env_ids, 4])
        if self.apply_waist_pitch_only_when_stance:
            self.zero_fix_waist_pitch[env_ids] *= (1 - self.commands[env_ids, 4])
        # if self.fix_waist_yaw:
        #     self.fixed_waist_yaw_pos[env_ids] = torch_rand_float(self.fix_waist_yaw_range[0], self.fix_waist_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze(1) * self.zero_fix_waist_yaw[env_ids]
        # if self.fix_waist_pitch:
        #     self.fixed_waist_pitch_pos[env_ids] = torch_rand_float(self.fix_waist_pitch_range[0], self.fix_waist_pitch_range[1], (len(env_ids), 1), device=self.device).squeeze(1) * self.zero_fix_waist_pitch[env_ids]
        # if self.fix_waist_roll:
        #     self.fixed_waist_roll_pos[env_ids] = torch_rand_float(self.fix_waist_roll_range[0], self.fix_waist_roll_range[1], (len(env_ids), 1), device=self.device).squeeze(1) * self.zero_fix_waist_roll[env_ids]
        self.commands[env_ids, 5] = torch_rand_float(self.fix_waist_yaw_range[0], self.fix_waist_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze(1) * self.zero_fix_waist_yaw[env_ids]
        self.commands[env_ids, 6] = torch_rand_float(self.fix_waist_roll_range[0], self.fix_waist_roll_range[1], (len(env_ids), 1), device=self.device).squeeze(1) * self.zero_fix_waist_roll[env_ids]
        self.commands[env_ids, 7] = torch_rand_float(self.fix_waist_pitch_range[0], self.fix_waist_pitch_range[1], (len(env_ids), 1), device=self.device).squeeze(1) * self.zero_fix_waist_pitch[env_ids]
        
    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        
        self.commands = torch.zeros((self.num_envs, 8), dtype=torch.float32, device=self.device)
        # Apply full upper body action scale
        self.action_scale_upper_body = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # TODO: haotian: adding command configuration
        if command is not None:
            self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands

        # self.config.obs.noise_scales = {
        #     key: value * 0.0 for key, value in self.config.obs.noise_scales.items()
        # }
        # print("noise scales: ", self.config.obs.noise_scales)

    ################ Curriculum #################
    
    ########################### FEET REWARDS ###########################
    
    ######################## LIMITS REWARDS #########################
    
    ######################### PENALTY REWARDS #########################
    
    def _reward_penalty_action_rate(self):
        # Penalize changes in actions (lower body only)
        return torch.sum(torch.square(self.last_actions[:, :self.config.robot.lower_body_actions_dim] - \
                                      self.actions[:, :self.config.robot.lower_body_actions_dim]), dim=1)
    
    def _reward_tracking_waist_dofs(self):
        # Penalize the difference between the waist dof pos and the reference
        waist_pos = self.simulator.dof_pos[:, self.waist_dof_indices]
        waist_dofs_error =  torch.sum(torch.square(waist_pos - self.commands[:, 5:8]), dim=1)
        return torch.exp(-waist_dofs_error/self.config.rewards.reward_tracking_sigma.waist_dofs)

    ######################### Observations #########################
    def _get_obs_command_waist_dofs(self):
        return self.commands[:, 5:8]
    
    def _get_obs_base_orientation(self):
        return self.base_quat[:, 0:4]