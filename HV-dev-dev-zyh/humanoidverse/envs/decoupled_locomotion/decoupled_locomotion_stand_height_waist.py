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
from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_waist import LeggedRobotDecoupledLocomotionStanceWaist
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
class LeggedRobotDecoupledLocomotionStanceHeightWaist(LeggedRobotDecoupledLocomotionStanceWaist):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)

    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 9), dtype=torch.float32, device=self.device
        )

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        # Sample the desired base height
        self.commands[env_ids, 8] = self.config.rewards.desired_base_height
        self.commands[env_ids, 8] += torch_rand_float(self.command_ranges["base_height"][0], 
                                                      self.command_ranges["base_height"][1], 
                                                      (len(env_ids), 1), device=self.device).squeeze(1) * (1.0 - self.commands[env_ids, 4]) # only apply the base height if standing
    
    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        
        self.commands = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)
        self.commands[:, 8] = self.config.rewards.desired_base_height
        # Apply full upper body action scale
        self.action_scale_upper_body = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # TODO: haotian: adding command configuration
        if command is not None:
            self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands

        # self.config.obs.noise_scales = {
        #     key: value * 0.0 for key, value in self.config.obs.noise_scales.items()
        # }
        # print("noise scales: ", self.config.obs.noise_scales)
    
    ########################### FEET REWARDS ###########################
    def _reward_penalty_hip_pos(self):
        # Penalize the hip joints (only roll and yaw)
        hips_roll_yaw_indices = self.hips_dof_id[1:3] + self.hips_dof_id[4:6]
        hip_pos = self.simulator.dof_pos[:, hips_roll_yaw_indices]
        penalty_hip_pos = torch.sum(torch.square(hip_pos), dim=1)
        return penalty_hip_pos * (self.commands[:, 4] + (1 - self.commands[:, 4]) * self.commands[:, 8])

    def _reward_penalty_torso_orientation(self):
        # Penalize non flat torso orientation
        torso_quat = self.simulator._rigid_body_rot[:, self.torso_index]
        projected_gravity_torso = quat_rotate_inverse(torso_quat, self.gravity_vec)
        torso_quat_target = quat_from_euler_xyz(self.commands[:, 6], self.commands[:, 7], self.commands[:, 5])
        projected_gravity_torso_target = quat_rotate_inverse(torso_quat_target, self.gravity_vec)
        projected_gravity_torso_error = projected_gravity_torso - projected_gravity_torso_target
        return torch.sum(torch.square(projected_gravity_torso_error[:, :2]), dim=1) * self.apply_waist_roll_pitch_only_when_stance
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.simulator.robot_root_states[:, 2]
        # return torch.square(base_height - self.config.rewards.desired_base_height)*self.commands[:, 4] # only apply the base height penalty if locomoting
        penalty_base_height = torch.square(base_height - self.commands[:, 8]) # only apply the base height penalty if standing
        stance_env_idx = torch.where(self.commands[:, 4] < 1)[0]
        penalty_base_height[stance_env_idx] *= 2.5 # double the penalty if standing
        return penalty_base_height
    
    def _reward_tracking_base_height(self):
        # Tracking of base height commands (z axe)
        base_height_error = torch.abs(self.commands[:, 8] - self.simulator.robot_root_states[:, 2])
        return torch.exp(-base_height_error/self.config.rewards.reward_tracking_sigma.base_height)*(1.0 - self.commands[:, 4]) # only apply the base height reward if standing
    
    ######################### Observations #########################
    def _get_obs_command_base_height(self):
        return self.commands[:, 8:9]