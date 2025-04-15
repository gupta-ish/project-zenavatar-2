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
from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_height_waist import LeggedRobotDecoupledLocomotionStanceHeightWaist
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
class LeggedRobotDecoupledLocomotionStanceHeightWaistForce(LeggedRobotDecoupledLocomotionStanceHeightWaist):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.left_hand_link = config.robot.force_control.left_hand_link
        self.right_hand_link = config.robot.force_control.right_hand_link
        self.left_hand_link_index = self.body_names.index(self.left_hand_link)
        self.right_hand_link_index = self.body_names.index(self.right_hand_link)
        logger.info(f"Left Hand Link: {self.left_hand_link}, Index: {self.left_hand_link_index}")
        logger.info(f"Right Hand Link: {self.right_hand_link}, Index: {self.right_hand_link_index}")
        self.force_xyz_scale = torch.distributions.Dirichlet(torch.tensor([1.0, 1.0, 1.0], device=self.device)).sample((self.num_envs, ))
        self.force_range_low = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_range_high = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_range_low[:, 0] = self.config.apply_force_x_range[0]; self.force_range_high[:, 0] = self.config.apply_force_x_range[1]
        self.force_range_low[:, 1] = self.config.apply_force_y_range[0]; self.force_range_high[:, 1] = self.config.apply_force_y_range[1]
        self.force_range_low[:, 2] = self.config.apply_force_z_range[0]; self.force_range_high[:, 2] = self.config.apply_force_z_range[1]
        
        self.apply_force_scale = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.left_ee_apply_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_apply_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_ankle_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.left_ankle_dof_names]
        self.right_ankle_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.right_ankle_dof_names]

        self.j_left_ee = torch.zeros((self.num_envs, 6, 6+self.num_dofs), device=self.device)
        self.j_right_ee = torch.zeros((self.num_envs, 6, 6+self.num_dofs), device=self.device)
        
        self.only_apply_z_force_when_walking = self.config.get("only_apply_z_force_when_walking", True)
        self.only_apply_resistance_when_walking = self.config.get("only_apply_resistance_when_walking", True)

    def _init_buffers(self):
        super()._init_buffers()
        self.apply_force_tensor = torch.zeros(self.num_envs, self.config.robot.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.apply_force_pos_tensor = torch.zeros(self.num_envs, self.config.robot.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # force ranges
        self.force_range_low = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_range_high = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_range_low[:] = self.config.apply_force_range[0]
        self.force_range_high[:] = self.config.apply_force_range[1]
        # force duration
        self.apply_force_duration = torch.randint(self.config.randomize_force_duration[0], self.config.randomize_force_duration[1] + 1, (self.num_envs, 1), device=self.device)
        self.left_ee_apply_force_phase = torch.rand((self.num_envs, 1), device=self.device)
        self.right_ee_apply_force_phase = torch.rand((self.num_envs, 1), device=self.device)
        self.left_ee_apply_force_phase_ts = torch.zeros((self.num_envs, 1), device=self.device)
        self.right_ee_apply_force_phase_ts = torch.zeros((self.num_envs, 1), device=self.device)
        # zero force probability
        self.left_zero_force_prob = self.config.get("zero_force_prob", [0.2, 0.2, 0.2])
        self.right_zero_force_prob = self.config.get("zero_force_prob", [0.2, 0.2, 0.2])
        if isinstance(self.left_zero_force_prob, float):
            self.left_zero_force_prob = [self.left_zero_force_prob] * 3
        if isinstance(self.right_zero_force_prob, float):
            self.right_zero_force_prob = [self.right_zero_force_prob] * 3
        self.left_zero_force_prob = torch.tensor(self.left_zero_force_prob, device=self.device)
        self.right_zero_force_prob = torch.tensor(self.right_zero_force_prob, device=self.device)
        self.left_zero_force = (torch.rand((self.num_envs, 3), device=self.device) < self.left_zero_force_prob).float()
        self.right_zero_force = (torch.rand((self.num_envs, 3), device=self.device) < self.right_zero_force_prob).float()
        # random force probability
        self.random_force_prob = self.config.get("random_force_prob", 0.0)
        self.random_force = (torch.rand((self.num_envs, 1), device=self.device) < self.random_force_prob).float()

        # stance/random force environments
        self.env_ids_stance = torch.where(self.commands[:, 4] == 0)[0]
        self.env_ids_stance_random = self.env_ids_stance[torch.where(self.random_force[self.env_ids_stance] == 1)[0]]
        self.env_ids_stance_non_random = self.env_ids_stance[torch.where(self.random_force[self.env_ids_stance] == 0)[0]]
        self.force_phase_ts_up_or_down = torch.ones((self.num_envs, 1), device=self.device) # 1 for up, -1 for down
    
    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        self._update_apply_force_phase()
    
    def _update_apply_force_phase(self):
        # For the stance environments, update the phase timestamp
        self.left_ee_apply_force_phase_ts[self.env_ids_stance] += 1.0 / self.apply_force_duration[self.env_ids_stance]
        self.right_ee_apply_force_phase_ts[self.env_ids_stance] += 1.0 / self.apply_force_duration[self.env_ids_stance]
        # self.apply_force_phase_ts[self.env_ids_stance] += self.force_phase_ts_up_or_down[self.env_ids_stance] / self.apply_force_duration[self.env_ids_stance]
        
        # For the non-random stance environments, update the phase continuously
        self.left_ee_apply_force_phase[self.env_ids_stance_non_random] = abs(torch.remainder(self.left_ee_apply_force_phase_ts[self.env_ids_stance_non_random], 2.0) - 1.0)
        self.right_ee_apply_force_phase[self.env_ids_stance_non_random] = abs(torch.remainder(self.right_ee_apply_force_phase_ts[self.env_ids_stance_non_random], 2.0) - 1.0)
        # For the random stance environments, update the phase randomly
        # left_ee_env_ids_stance_random_resample = self.env_ids_stance_random[torch.where(self.left_ee_apply_force_phase_ts[self.env_ids_stance_random] >= 1.0)[0]]
        # right_ee_env_ids_stance_random_resample = self.env_ids_stance_random[torch.where(self.right_ee_apply_force_phase_ts[self.env_ids_stance_random] >= 1.0)[0]]
        # self.left_ee_apply_force_phase[left_ee_env_ids_stance_random_resample] = torch.rand((len(left_ee_env_ids_stance_random_resample), 1), device=self.device)
        # self.right_ee_apply_force_phase[right_ee_env_ids_stance_random_resample] = torch.rand((len(right_ee_env_ids_stance_random_resample), 1), device=self.device)
        # self.left_ee_apply_force_phase_ts[left_ee_env_ids_stance_random_resample] = 0.0
        # self.right_ee_apply_force_phase_ts[right_ee_env_ids_stance_random_resample] = 0.0
        # print(f"Apply Force Phase: {self.apply_force_phase}", " Apply Force Duration: ", self.apply_force_duration)
    
    # def _calculate_max_ee_force(self):
    #     # Apply the force at the hand links
    #     # Compute the linear z-axis jacobian for the left and right hand links
    #     jacobian = self.simulator.jacobian # Shape: (num_envs, num_bodies, 6, 35)
    #     # import ipdb; ipdb.set_trace()
    #     self.j_left_ee = jacobian[:, self.left_hand_link_index, :, :] # Shape: (num_envs, 6, 35)
    #     self.j_right_ee = jacobian[:, self.right_hand_link_index, :, :] # Shape: (num_envs, 6, 35)
    #     # i = self.body_names.index("left_rubber_hand"); j = self.body_names.index("right_rubber_hand")
    #     # Ji = jacobian[:, i, :, :] # Shape: (num_envs, 6, 35)
    #     # Jj = jacobian[:, j, :, :] # Shape: (num_envs, 6, 35)
    #     # print(f"j_left_ee_upper_joint_linear: {Ji[:, 2, -self.num_upper_dofs:]}") # Shape: (num_envs, 1, 14)
    #     # print(f"j_right_ee_upper_joint_linear: {Jj[:, 2, -self.num_upper_dofs:]}") # Shape: (num_envs, 1, 14)
    #     j_left_ee_joint_linear = self.j_left_ee[:, :3, 6:]  # Shape: (num_envs, 3, 29)
    #     j_right_ee_joint_linear = self.j_right_ee[:, :3, 6:]  # Shape: (num_envs, 3, 29)
    #     inv_sum_j_ee_joint_linear = 1.0 / (j_left_ee_joint_linear + j_right_ee_joint_linear + 1e-3) # Shape: (num_envs, 3, 17)
    #     inv_sum_j_ee_joint_linear_x = inv_sum_j_ee_joint_linear[:, 0, :]  # Shape: (num_envs, 17)
    #     inv_sum_j_ee_joint_linear_y = inv_sum_j_ee_joint_linear[:, 1, :]  # Shape: (num_envs, 17)
    #     inv_sum_j_ee_joint_linear_z = inv_sum_j_ee_joint_linear[:, 2, :]  # Shape: (num_envs, 17)
    #     joint_effort_limit = torch.tensor(self.config.robot.dof_effort_limit_list[:], 
    #                                       device=self.device, requires_grad=False)
    #     joint_effort_est = self.simulator.dof_forces[:, :]
    #     max_delta_joint_effort = joint_effort_limit - joint_effort_est
    #     min_delta_joint_effort = -joint_effort_limit - joint_effort_est
        
    #     # Yuanhang: Apply X-Y-Z forces to the stance env, and apply only Z force to the walking env
    #     ee_force_min = torch.zeros((self.num_envs, 3), device=self.device)
    #     ee_force_max = torch.zeros((self.num_envs, 3), device=self.device)
        
    #     if self.only_apply_z_force_when_walking:
    #         stance_envs_idx = torch.where(self.commands[:, 4] == 0)[0]
    #         walk_envs_idx = torch.where(self.commands[:, 4] == 1)[0]
    #         if len(stance_envs_idx) > 0:
    #             idx = stance_envs_idx
    #             ee_force_max_delta_joint_x = inv_sum_j_ee_joint_linear_x[idx] * max_delta_joint_effort[idx] # Shape: (num_envs, 29)
    #             ee_force_min_delta_joint_x = inv_sum_j_ee_joint_linear_x[idx] * min_delta_joint_effort[idx]
    #             ee_force_max_delta_joint_y = inv_sum_j_ee_joint_linear_y[idx] * max_delta_joint_effort[idx]
    #             ee_force_min_delta_joint_y = inv_sum_j_ee_joint_linear_y[idx] * min_delta_joint_effort[idx]
    #             ee_force_max_delta_joint_z = inv_sum_j_ee_joint_linear_z[idx] * max_delta_joint_effort[idx]
    #             ee_force_min_delta_joint_z = inv_sum_j_ee_joint_linear_z[idx] * min_delta_joint_effort[idx]
    #             ee_force_min_x = torch.max(torch.min(ee_force_max_delta_joint_x, ee_force_min_delta_joint_x), dim=1)[0].unsqueeze(1) # Shape: (num_envs, 1)
    #             ee_force_min_y = torch.max(torch.min(ee_force_max_delta_joint_y, ee_force_min_delta_joint_y), dim=1)[0].unsqueeze(1) 
    #             ee_force_min_z = torch.max(torch.min(ee_force_max_delta_joint_z, ee_force_min_delta_joint_z), dim=1)[0].unsqueeze(1) 
    #             ee_force_max_x = torch.min(torch.max(ee_force_max_delta_joint_x, ee_force_min_delta_joint_x), dim=1)[0].unsqueeze(1) # Shape: (num_envs, 1)
    #             ee_force_max_y = torch.min(torch.max(ee_force_max_delta_joint_y, ee_force_min_delta_joint_y), dim=1)[0].unsqueeze(1) 
    #             ee_force_max_z = torch.min(torch.max(ee_force_max_delta_joint_z, ee_force_min_delta_joint_z), dim=1)[0].unsqueeze(1) 
    #             # Apply the force scale and the force phase to the maximum force norm
    #             ee_force_min[idx] = torch.cat([ee_force_min_x * self.force_xyz_scale[idx, 0:1], 
    #                                         ee_force_min_y * self.force_xyz_scale[idx, 1:2], 
    #                                         ee_force_min_z * self.force_xyz_scale[idx, 2:3]], dim=1)
    #             ee_force_max[idx] = torch.cat([ee_force_max_x * self.force_xyz_scale[idx, 0:1],
    #                                         ee_force_max_y * self.force_xyz_scale[idx, 1:2],
    #                                         ee_force_max_z * self.force_xyz_scale[idx, 2:3]], dim=1)
    #         if len(walk_envs_idx) > 0:
    #             idx = walk_envs_idx
    #             ee_force_max_delta_joint_z = inv_sum_j_ee_joint_linear_z[idx] * max_delta_joint_effort[idx] # Shape: (num_envs, 29)
    #             ee_force_min_delta_joint_z = inv_sum_j_ee_joint_linear_z[idx] * min_delta_joint_effort[idx]
    #             ee_force_min_z = torch.max(torch.min(ee_force_max_delta_joint_z, ee_force_min_delta_joint_z), dim=1)[0].unsqueeze(1) # Shape: (num_envs, 1)
    #             ee_force_max_z = torch.min(torch.max(ee_force_max_delta_joint_z, ee_force_min_delta_joint_z), dim=1)[0].unsqueeze(1) # Shape: (num_envs, 1)
    #             ee_force_max[idx] = torch.cat([torch.zeros_like(ee_force_max_z), torch.zeros_like(ee_force_max_z), ee_force_max_z], dim=1)
    #             ee_force_min[idx] = torch.cat([torch.zeros_like(ee_force_min_z), torch.zeros_like(ee_force_min_z), ee_force_min_z], dim=1)
    #     else:
    #         idx = torch.arange(self.num_envs)
    #         ee_force_max_delta_joint_x = inv_sum_j_ee_joint_linear_x[idx] * max_delta_joint_effort[idx] # Shape: (num_envs, 29)
    #         ee_force_min_delta_joint_x = inv_sum_j_ee_joint_linear_x[idx] * min_delta_joint_effort[idx]
    #         ee_force_max_delta_joint_y = inv_sum_j_ee_joint_linear_y[idx] * max_delta_joint_effort[idx]
    #         ee_force_min_delta_joint_y = inv_sum_j_ee_joint_linear_y[idx] * min_delta_joint_effort[idx]
    #         ee_force_max_delta_joint_z = inv_sum_j_ee_joint_linear_z[idx] * max_delta_joint_effort[idx]
    #         ee_force_min_delta_joint_z = inv_sum_j_ee_joint_linear_z[idx] * min_delta_joint_effort[idx]
    #         ee_force_min_x = torch.max(torch.min(ee_force_max_delta_joint_x, ee_force_min_delta_joint_x), dim=1)[0].unsqueeze(1) # Shape: (num_envs, 1)
    #         ee_force_min_y = torch.max(torch.min(ee_force_max_delta_joint_y, ee_force_min_delta_joint_y), dim=1)[0].unsqueeze(1) 
    #         ee_force_min_z = torch.max(torch.min(ee_force_max_delta_joint_z, ee_force_min_delta_joint_z), dim=1)[0].unsqueeze(1) 
    #         ee_force_max_x = torch.min(torch.max(ee_force_max_delta_joint_x, ee_force_min_delta_joint_x), dim=1)[0].unsqueeze(1) # Shape: (num_envs, 1)
    #         ee_force_max_y = torch.min(torch.max(ee_force_max_delta_joint_y, ee_force_min_delta_joint_y), dim=1)[0].unsqueeze(1) 
    #         ee_force_max_z = torch.min(torch.max(ee_force_max_delta_joint_z, ee_force_min_delta_joint_z), dim=1)[0].unsqueeze(1) 
    #         # Apply the force scale and the force phase to the maximum force norm
    #         ee_force_min[idx] = torch.cat([ee_force_min_x * self.force_xyz_scale[idx, 0:1], 
    #                                         ee_force_min_y * self.force_xyz_scale[idx, 1:2], 
    #                                         ee_force_min_z * self.force_xyz_scale[idx, 2:3]], dim=1)
    #         ee_force_max[idx] = torch.cat([ee_force_max_x * self.force_xyz_scale[idx, 0:1],
    #                                         ee_force_max_y * self.force_xyz_scale[idx, 1:2],
    #                                         ee_force_max_z * self.force_xyz_scale[idx, 2:3]], dim=1)
    #     # if self.is_evaluating: print(f"Max EE Force, Z: {ee_force_max_z}")
    #     # print(f"Max EE Force: {ee_force_max}, Min EE Force: {ee_force_min}, Force XYZ Scale: {self.force_xyz_scale}")
    #     # Compute phased forces
    #     # TODO [Yuanhang]: apply different force phase to left and right EE
    #     left_ee_force_phased = ee_force_min + (ee_force_max - ee_force_min) * self.left_ee_apply_force_phase
    #     right_ee_force_phased = ee_force_min + (ee_force_max - ee_force_min) * self.right_ee_apply_force_phase

    #     # Apply the force scale and the force phase to the maximum force norm
    #     left_force_max = left_ee_force_phased * self.apply_force_scale + torch.rand((self.num_envs, 3), device=self.device) # Shape: (num_envs, 3)
    #     right_force_max = right_ee_force_phased * self.apply_force_scale + torch.rand((self.num_envs, 3), device=self.device) # Shape: (num_envs, 3)
    #     # import ipdb; ipdb.set_trace()
        
    #     # Zero the force if zero force probability is met
    #     left_force_max *= (1 - self.left_zero_force)
    #     right_force_max *= (1 - self.right_zero_force)

    #     # Clip the force norm to the range [force_range_low, force_range_high] and scale it by the random force scale
    #     left_hand_force = torch.clip(left_force_max, self.force_range_low, self.force_range_high) # Shape: (num_envs, 3)
    #     right_hand_force = torch.clip(right_force_max, self.force_range_low, self.force_range_high) # Shape: (num_envs, 3)

    #     return left_hand_force, right_hand_force
    
    def _calculate_max_ee_force(self):
        # Apply the force at the hand links
        # Compute the linear z-axis jacobian for the left and right hand links
        jacobian = self.simulator.jacobian # Shape: (num_envs, num_bodies, 6, 35)
        # import ipdb; ipdb.set_trace()
        self.j_left_ee = jacobian[:, self.left_hand_link_index, :, :] # Shape: (num_envs, 6, 35)
        self.j_right_ee = jacobian[:, self.right_hand_link_index, :, :] # Shape: (num_envs, 6, 35)
        # i = self.body_names.index("left_rubber_hand"); j = self.body_names.index("right_rubber_hand")
        # Ji = jacobian[:, i, :, :] # Shape: (num_envs, 6, 35)
        # Jj = jacobian[:, j, :, :] # Shape: (num_envs, 6, 35)
        # print(f"j_left_ee_upper_joint_linear: {Ji[:, 2, -self.num_upper_dofs:]}") # Shape: (num_envs, 1, 14)
        # print(f"j_right_ee_upper_joint_linear: {Jj[:, 2, -self.num_upper_dofs:]}") # Shape: (num_envs, 1, 14)
        j_left_ee_joint_linear = self.j_left_ee[:, :3, 6:]  # Shape: (num_envs, 3, 29)
        j_right_ee_joint_linear = self.j_right_ee[:, :3, 6:]  # Shape: (num_envs, 3, 29)
        j_left_ee_arm_joint_linear = j_left_ee_joint_linear[:, :, self.left_arm_dof_indices]  # Shape: (num_envs, 3, 7)
        j_right_ee_arm_joint_linear = j_right_ee_joint_linear[:, :, self.right_arm_dof_indices]  # Shape: (num_envs, 3, 7)
        j_left_ee_waist_joint_linear = j_left_ee_joint_linear[:, :, self.waist_dof_indices]  # Shape: (num_envs, 3, 3)
        j_right_ee_waist_joint_linear = j_right_ee_joint_linear[:, :, self.waist_dof_indices]  # Shape: (num_envs, 3, 3)
        joint_effort_limit = torch.tensor(self.config.robot.dof_effort_limit_list[:], 
                                          device=self.device, requires_grad=False)
        joint_effort_est = self.simulator.dof_forces[:, :]
        max_delta_joint_effort = joint_effort_limit - joint_effort_est
        min_delta_joint_effort = -joint_effort_limit - joint_effort_est
        
        # Yuanhang: Apply X-Y-Z forces to the stance env, and apply only Z force to the walking env
        left_ee_force_min = torch.zeros((self.num_envs, 3), device=self.device)
        left_ee_force_max = torch.zeros((self.num_envs, 3), device=self.device)
        right_ee_force_min = torch.zeros((self.num_envs, 3), device=self.device)
        right_ee_force_max = torch.zeros((self.num_envs, 3), device=self.device)
        
        left_ee_force_max_delta_joint = torch.mul((1.0 / (j_left_ee_arm_joint_linear + 1e-3)), torch.stack([max_delta_joint_effort[:, self.left_arm_dof_indices]] * 3, dim=1)) # Shape: (num_envs, 3, 7)
        left_ee_force_min_delta_joint = torch.mul((1.0 / (j_left_ee_arm_joint_linear + 1e-3)), torch.stack([min_delta_joint_effort[:, self.left_arm_dof_indices]] * 3, dim=1))
        right_ee_force_max_delta_joint = torch.mul((1.0 / (j_right_ee_arm_joint_linear + 1e-3)), torch.stack([max_delta_joint_effort[:, self.right_arm_dof_indices]] * 3, dim=1)) # Shape: (num_envs, 3, 7)
        right_ee_force_min_delta_joint = torch.mul((1.0 / (j_right_ee_arm_joint_linear + 1e-3)), torch.stack([min_delta_joint_effort[:, self.right_arm_dof_indices]] * 3, dim=1))
        left_ee_force_min = torch.max(torch.min(left_ee_force_max_delta_joint, left_ee_force_min_delta_joint), dim=2)[0]
        left_ee_force_max = torch.min(torch.max(left_ee_force_max_delta_joint, left_ee_force_min_delta_joint), dim=2)[0]
        right_ee_force_min = torch.max(torch.min(right_ee_force_max_delta_joint, right_ee_force_min_delta_joint), dim=2)[0]
        right_ee_force_max = torch.min(torch.max(right_ee_force_max_delta_joint, right_ee_force_min_delta_joint), dim=2)[0]
        left_ee_force_min = torch.cat([left_ee_force_min[:, 0:1] * self.force_xyz_scale[:, 0:1],
                                       left_ee_force_min[:, 1:2] * self.force_xyz_scale[:, 1:2],
                                       left_ee_force_min[:, 2:3] * self.force_xyz_scale[:, 2:3]], dim=1)
        left_ee_force_max = torch.cat([left_ee_force_max[:, 0:1] * self.force_xyz_scale[:, 0:1],
                                       left_ee_force_max[:, 1:2] * self.force_xyz_scale[:, 1:2],
                                       left_ee_force_max[:, 2:3] * self.force_xyz_scale[:, 2:3]], dim=1)
        right_ee_force_min = torch.cat([right_ee_force_min[:, 0:1] * self.force_xyz_scale[:, 0:1],
                                        right_ee_force_min[:, 1:2] * self.force_xyz_scale[:, 1:2],
                                        right_ee_force_min[:, 2:3] * self.force_xyz_scale[:, 2:3]], dim=1)
        right_ee_force_max = torch.cat([right_ee_force_max[:, 0:1] * self.force_xyz_scale[:, 0:1],
                                        right_ee_force_max[:, 1:2] * self.force_xyz_scale[:, 1:2],
                                        right_ee_force_max[:, 2:3] * self.force_xyz_scale[:, 2:3]], dim=1)
        
        if self.only_apply_z_force_when_walking:
            walk_envs_idx = torch.where(self.commands[:, 4] == 1)[0]
            if len(walk_envs_idx) > 0:
                idx = walk_envs_idx
                left_ee_force_max[idx] = torch.cat([torch.zeros_like(left_ee_force_max[idx, 0:1]),
                                                    torch.zeros_like(left_ee_force_max[idx, 1:2]),
                                                    left_ee_force_max[idx, 2:3]], dim=1)
                left_ee_force_min[idx] = torch.cat([torch.zeros_like(left_ee_force_min[idx, 0:1]),
                                                    torch.zeros_like(left_ee_force_min[idx, 1:2]),
                                                    left_ee_force_min[idx, 2:3]], dim=1)
                right_ee_force_max[idx] = torch.cat([torch.zeros_like(right_ee_force_max[idx, 0:1]),
                                                     torch.zeros_like(right_ee_force_max[idx, 1:2]),
                                                     right_ee_force_max[idx, 2:3]], dim=1)
                right_ee_force_min[idx] = torch.cat([torch.zeros_like(right_ee_force_min[idx, 0:1]),
                                                     torch.zeros_like(right_ee_force_min[idx, 1:2]),
                                                     right_ee_force_min[idx, 2:3]], dim=1)

        # if self.is_evaluating: print(f"Max EE Force, Z: {ee_force_max_z}")
        # print(f"Max EE Force: {ee_force_max}, Min EE Force: {ee_force_min}, Force XYZ Scale: {self.force_xyz_scale}")
        # Compute phased forces
        # TODO [Yuanhang]: apply different force phase to left and right EE
        left_ee_force_phased = left_ee_force_min + (left_ee_force_max - left_ee_force_min) * self.left_ee_apply_force_phase
        right_ee_force_phased = right_ee_force_min + (right_ee_force_max - right_ee_force_min) * self.right_ee_apply_force_phase

        # Apply the force scale and the force phase to the maximum force norm
        left_force_max = left_ee_force_phased * self.apply_force_scale + torch.rand((self.num_envs, 3), device=self.device) # Shape: (num_envs, 3)
        right_force_max = right_ee_force_phased * self.apply_force_scale + torch.rand((self.num_envs, 3), device=self.device) # Shape: (num_envs, 3)
        # import ipdb; ipdb.set_trace()
        
        # Zero the force if zero force probability is met
        left_force_max *= (1 - self.left_zero_force)
        right_force_max *= (1 - self.right_zero_force)

        # Clip the force norm to the range [force_range_low, force_range_high] and scale it by the random force scale
        left_hand_force = torch.clip(left_force_max, self.force_range_low, self.force_range_high) # Shape: (num_envs, 3)
        right_hand_force = torch.clip(right_force_max, self.force_range_low, self.force_range_high) # Shape: (num_envs, 3)
        
        left_ee_torque_on_waist = j_left_ee_waist_joint_linear.transpose(1, 2).bmm(left_hand_force.unsqueeze(-1)).squeeze(-1)
        right_ee_torque_on_waist = j_right_ee_waist_joint_linear.transpose(1, 2).bmm(right_hand_force.unsqueeze(-1)).squeeze(-1)
        waist_torque_limit = joint_effort_limit[self.waist_dof_indices]
        # import ipdb; ipdb.set_trace()
        total_waist_torque = left_ee_torque_on_waist + right_ee_torque_on_waist  
        scaling_factor = torch.min(torch.ones_like(total_waist_torque), waist_torque_limit / (total_waist_torque.abs() + 1e-6))
        if scaling_factor.min() < 1.0 and self.is_evaluating:
            print(f"Scaling Factor: {scaling_factor.min()}")
            # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        left_hand_force *= scaling_factor
        right_hand_force *= scaling_factor

        return left_hand_force, right_hand_force

    def _apply_force_compensation_in_physics_step(self, left_hand_force, right_hand_force):
        # Calculate torque compensation
        j_left_ee_lower_joint_linear = self.j_left_ee[:, :3, 6:6+self.num_lower_dofs]  # Shape: (num_envs, 3, 12)
        j_left_ee_upper_joint_linear = self.j_left_ee[:, :3, 6+self.num_lower_dofs:]  # Shape: (num_envs, 3, 17)
        j_right_ee_lower_joint_linear = self.j_right_ee[:, :3, 6:6+self.num_lower_dofs]  # Shape: (num_envs, 3, 12)
        j_right_ee_upper_joint_linear = self.j_right_ee[:, :3, 6+self.num_lower_dofs:]  # Shape: (num_envs, 3, 17)
        # Compute the feed forward torques for upper dofs
        left_ee_upper_dof_compensation = torch.bmm(j_left_ee_upper_joint_linear.transpose(1, 2), 
                                                    left_hand_force.unsqueeze(2)).squeeze(2)
        right_ee_upper_dof_compensation = torch.bmm(j_right_ee_upper_joint_linear.transpose(1, 2),
                                                    right_hand_force.unsqueeze(2)).squeeze(2)
        
        # TODO [Yuanhang]: Apply different force compensation to left and right EE
        self.left_ee_comp = left_ee_upper_dof_compensation.clone()
        self.right_ee_comp = right_ee_upper_dof_compensation.clone()
        
        feed_forward_upper_dof_torques = left_ee_upper_dof_compensation + right_ee_upper_dof_compensation
        # Compute the feed forward torques for lower dofs
        if self.config.get("lower_body_force_compensation", False):
            left_ee_lower_dof_compensation = torch.bmm(j_left_ee_lower_joint_linear.transpose(1, 2),
                                                        left_hand_force.unsqueeze(2)).squeeze(2)
            right_ee_lower_dof_compensation = torch.bmm(j_right_ee_lower_joint_linear.transpose(1, 2),
                                                        right_hand_force.unsqueeze(2)).squeeze(2)
            feed_forward_lower_dof_torques = left_ee_lower_dof_compensation + right_ee_lower_dof_compensation
            self.left_ee_comp = torch.cat([left_ee_lower_dof_compensation, left_ee_upper_dof_compensation], dim=1)
            self.right_ee_comp = torch.cat([right_ee_lower_dof_compensation, right_ee_upper_dof_compensation], dim=1)
            feed_forward_torques = torch.concat([feed_forward_lower_dof_torques, feed_forward_upper_dof_torques], dim=1)
        else:
            feed_forward_torques = torch.concat([torch.zeros((self.num_envs, self.num_lower_dofs), device=self.device), feed_forward_upper_dof_torques], dim=1)
        self.torques -= feed_forward_torques
    
    def _apply_force_in_physics_step(self):
        # Apply the force in the physics step
        self.torques = self._compute_torques(self.actions_after_delay).view(self.torques.shape)
        
        left_hand_force, right_hand_force = self._calculate_max_ee_force()

        if self.is_evaluating:
            # left_hand_force *= 0.0
            # right_hand_force *= 0.0
            pass
        
        # Get the walking direction
        walking_dir = quat_rotate(self.base_quat, 
                                  torch.cat([self.commands[:, :2], 
                                             torch.zeros((self.num_envs, 1), device=self.device)], 
                                             dim=-1))[:, :2]  # (batch_size, 2)
        walking_dir_norm = torch.norm(walking_dir, dim=-1, keepdim=True) + 1e-6 
        env_mask = (walking_dir_norm > 0.1).squeeze(-1)  # (batch_size,)

        if self.only_apply_resistance_when_walking:
            walking_dir_unit = torch.zeros_like(walking_dir)
            walking_dir_unit[env_mask] = -walking_dir[env_mask] / walking_dir_norm[env_mask]
            
            # Get the x-y components of the force
            left_hand_force_xy = left_hand_force[:, :2]  
            right_hand_force_xy = right_hand_force[:, :2]

            left_hand_force_proj = torch.zeros_like(left_hand_force_xy)
            right_hand_force_proj = torch.zeros_like(right_hand_force_xy)

            left_hand_force_proj[env_mask] = torch.abs(torch.sum(left_hand_force_xy[env_mask] * walking_dir_unit[env_mask], dim=-1, keepdim=True)) * walking_dir_unit[env_mask]
            right_hand_force_proj[env_mask] = torch.abs(torch.sum(right_hand_force_xy[env_mask] * walking_dir_unit[env_mask], dim=-1, keepdim=True)) * walking_dir_unit[env_mask]

            # Combine the x-y components with the z component
            left_hand_force = torch.cat([left_hand_force_proj, left_hand_force[:, 2:3]], dim=-1)
            right_hand_force = torch.cat([right_hand_force_proj, right_hand_force[:, 2:3]], dim=-1)

        self.left_ee_apply_force = quat_rotate_inverse(self.base_quat, left_hand_force.clone())
        self.right_ee_apply_force = quat_rotate_inverse(self.base_quat, right_hand_force.clone())

        # Apply the force to the hand links
        self.apply_force_tensor[:, self.left_hand_link_index, :] = left_hand_force
        self.apply_force_tensor[:, self.right_hand_link_index, :] = right_hand_force
        
        if self.config.apply_force_in_physics_step:
            self.simulator.apply_rigid_body_force_at_pos_tensor(self.apply_force_tensor, self.apply_force_pos_tensor)
        
        if self.config.apply_force_compensation_in_physics_step:
            self._apply_force_compensation_in_physics_step(left_hand_force, right_hand_force)
        
        self.simulator.apply_torques_at_dof(self.torques)
    
    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        # Yuanhang: use Dirichlet distribution to sample the force at maximum and minimum
        if env_ids.numel() > 0:  # Only update if env_ids is not empty
            self.force_xyz_scale[env_ids] = torch.distributions.Dirichlet(
                torch.tensor([1.0, 1.0, 1.0], device=self.device)
            ).sample((len(env_ids), ))
        
        self.apply_force_duration[env_ids] = torch.randint(self.config.randomize_force_duration[0], 
                                                           self.config.randomize_force_duration[1] + 1, 
                                                           (len(env_ids), 1), device=self.device)
        self.left_ee_apply_force_phase_ts[env_ids] = torch.rand((len(env_ids), 1), device=self.device)
        self.right_ee_apply_force_phase_ts[env_ids] = torch.rand((len(env_ids), 1), device=self.device)
        self.left_ee_apply_force_phase[env_ids] = torch.rand((len(env_ids), 1), device=self.device)
        self.right_ee_apply_force_phase[env_ids] = torch.rand((len(env_ids), 1), device=self.device)
        self.left_zero_force[env_ids] = (torch.rand((len(env_ids), 3), device=self.device) < self.left_zero_force_prob).float()
        self.right_zero_force[env_ids] = (torch.rand((len(env_ids), 3), device=self.device) < self.right_zero_force_prob).float()
        self.random_force[env_ids] = (torch.rand((len(env_ids), 1), device=self.device) < self.random_force_prob).float()

        if len(env_ids) != 0:
            self.env_ids_stance = torch.where(self.commands[env_ids, 4] == 0)[0]
            self.env_ids_stance_random = self.env_ids_stance[torch.where(self.random_force[self.env_ids_stance] == 1)[0]]
            self.env_ids_stance_non_random = self.env_ids_stance[torch.where(self.random_force[self.env_ids_stance] == 0)[0]]
            # For the random stance environments, update the phase randomly
            self.left_ee_apply_force_phase[self.env_ids_stance_random] = torch.rand((len(self.env_ids_stance_random), 1), device=self.device)
            self.right_ee_apply_force_phase[self.env_ids_stance_random] = torch.rand((len(self.env_ids_stance_random), 1), device=self.device)
            self.force_phase_ts_up_or_down[env_ids] = (torch.rand((len(env_ids), 1), device=self.device) < 0.5).float() * 2.0 - 1.0 # range: [-1, 1]
        
    def _update_force_scale_curriculum(self, env_ids):
        """ Implement force curriculum

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        env_ids_scale_up_mask = self.episode_length_buf[env_ids] > self.config.rewards.force_scale_up_threshold
        env_ids_scale_up = env_ids[torch.where(env_ids_scale_up_mask)[0]]
        env_ids_scale_down_mask = self.episode_length_buf[env_ids] < self.config.rewards.force_scale_down_threshold
        env_ids_scale_down = env_ids[torch.where(env_ids_scale_down_mask)[0]]
        self.apply_force_scale[env_ids_scale_up] += self.config.rewards.force_scale_up
        self.apply_force_scale[env_ids_scale_down] -= self.config.rewards.force_scale_down
        # Clip the scale
        self.apply_force_scale[env_ids] = torch.clip(self.apply_force_scale[env_ids], 
                                               self.config.rewards.force_scale_min, 
                                               self.config.rewards.force_scale_max)

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """ Resets the environments with the given ids and optionally to the target states
        """
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # if self.config.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
        #     self.update_command_curriculum(env_ids)
        if self.config.rewards.get("upper_body_motion_scale_curriculum", False):
            self._update_upper_body_motion_scale_curriculum(env_ids)
        if self.config.rewards.get("force_scale_curriculum", False):
            self._update_force_scale_curriculum(env_ids)
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

    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        self.debug_viz = True
        self.commands = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)
        self.commands[:, 8] = self.config.rewards.desired_base_height
        # Apply full upper body action scale
        self.action_scale_upper_body = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # Apply full force scale
        self.apply_force_scale = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # TODO: haotian: adding command configuration
        if command is not None:
            self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands
        
        # stance/random force environments
        self.env_ids_stance = torch.where(self.commands[:, 4] == 0)[0]
        self.env_ids_stance_random = self.env_ids_stance[torch.where(self.random_force[self.env_ids_stance] == 1)[0]]
        self.env_ids_stance_non_random = self.env_ids_stance[torch.where(self.random_force[self.env_ids_stance] == 0)[0]]
        self.force_phase_ts_up_or_down = torch.ones((self.num_envs, 1), device=self.device) # 1 for up, -1 for down
        
        # TODO [Yuanhang]: hardcode
        self.only_apply_z_force_when_walking = False
        self.only_apply_resistance_when_walking = True
        # self.config.obs.noise_scales = {
        #     key: value * 0.0 for key, value in self.config.obs.noise_scales.items()
        # }
        # print("noise scales: ", self.config.obs.noise_scales)
    
    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        ################### EXTEND Rigid body POS #####################
        rotated_pos_in_parent = my_quat_rotate(
            self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_pos_in_parent.reshape(-1, 3)
        )
        extend_curr_pos = my_quat_rotate(
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            rotated_pos_in_parent
        ).view(self.num_envs, -1, 3) + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
        self._rigid_body_pos_extend = torch.cat([self.simulator._rigid_body_pos, extend_curr_pos], dim=1)
        # Apply force at the hand links
        self.apply_force_pos_tensor[:, self.left_hand_link_index,:] = self.simulator._rigid_body_pos[:, self.left_hand_link_index, :]
        self.apply_force_pos_tensor[:, self.right_hand_link_index,:] = self.simulator._rigid_body_pos[:, self.right_hand_link_index, :]
        if self.is_evaluating:
            # self.ref_upper_dof_pos *= 0.0
            # self.ref_upper_dof_pos[:, 4] = 0.3
            # self.ref_upper_dof_pos[:, 11] = -0.3
            # self.ref_upper_dof_pos[:, 6] = 0.8
            # self.ref_upper_dof_pos[:, 13] = 0.8
            pass
    
    def _draw_debug_vis(self):
        self.simulator.clear_lines()
        self._refresh_sim_tensors()

        for env_id in range(self.num_envs):
            # for pos_id, pos_joint in enumerate(self.marker_coords[env_id]): # idx 0 torso (duplicate with 11)
            #     if self.config.robot.motion.visualization.customize_color:
            #         color_inner = self.config.robot.motion.visualization.marker_joint_colors[pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)]
            #     else:
            #         color_inner = (0.3, 0.3, 0.3)
            #     color_inner = tuple(color_inner)

            #     self.simulator.draw_sphere(pos_joint, 0.04, color_inner, env_id)
            # draw forces

            force_left_hand = self.apply_force_tensor[env_id, self.left_hand_link_index, :]
            force_right_hand = self.apply_force_tensor[env_id, self.right_hand_link_index, :]

            force_pos_left_hand = self.apply_force_pos_tensor[env_id, self.left_hand_link_index, :]
            force_pos_right_hand = self.apply_force_pos_tensor[env_id, self.right_hand_link_index, :]
            
            force_list = [force_left_hand, force_right_hand]
            pos_list = [force_pos_left_hand, force_pos_right_hand]
            force_mag_list = [0.025, 0.025]
            color_schems = [(0.851, 0.144, 0.07), (0.851, 0.144, 0.07)]
            # color_schems = [(0., .5, 0.), (0., .5, 0.)]
            line_widths = [0.02, 0.02]

            for force, pos, force_mag, color, line_width in zip(force_list, pos_list, force_mag_list, color_schems, line_widths):
                for _ in range(20):
                    start_point = pos + torch.rand(3, device=self.device) * line_width
                    end_point = pos + force * force_mag
                    self.simulator.draw_line(Point(start_point +torch.rand(3, device=self.device) * line_width),
                                        Point(end_point + torch.rand(3, device=self.device) * line_width),
                                        Point(color),
                                        env_id)

    ########################### FEET REWARDS ###########################
    def _reward_penalty_ankle_roll(self):
        # Compute the penalty for ankle roll
        left_ankle_roll = self.simulator.dof_pos[:, self.left_ankle_dof_indices[1:2]]
        right_ankle_roll = self.simulator.dof_pos[:, self.right_ankle_dof_indices[1:2]]
        return torch.sum(torch.square(left_ankle_roll) + torch.square(right_ankle_roll), dim=1)
    
    def _reward_penalty_stance_feet_vel(self):
        # Penalize the velocity of the stance feet
        left_ee_lin_vel = self.simulator._rigid_body_vel[:, self.left_hand_link_index, 0:3]
        left_ee_ang_vel = self.simulator._rigid_body_ang_vel[:, self.left_hand_link_index, 0:3]
        left_ee_vel = torch.cat([left_ee_lin_vel, left_ee_ang_vel], dim=1)
        right_ee_lin_vel = self.simulator._rigid_body_vel[:, self.right_hand_link_index, 0:3]
        right_ee_ang_vel = self.simulator._rigid_body_ang_vel[:, self.right_hand_link_index, 0:3]
        right_ee_vel = torch.cat([right_ee_lin_vel, right_ee_ang_vel], dim=1)
        return (torch.norm(left_ee_vel, dim=1) + torch.norm(right_ee_vel, dim=1)) * (1 - self.commands[:, 4])

    ######################### Observations #########################
    def _get_obs_left_ee_apply_force(self):
        # return the force exerted on the left ee (hand)
        # print(f"Apply Force Tensor: {self.apply_force_tensor[:, self.left_hand_link_index, 2:3]}")
        return self.left_ee_apply_force
    
    def _get_obs_right_ee_apply_force(self):
        # return the force exerted on the right ee (hand)
        return self.right_ee_apply_force
    
    def _get_obs_apply_force(self):
        # return the average force between the left and right hands
        # print(f"Apply Force Tensor: {self.apply_force_tensor[:, self.left_hand_link_index, 2:3]}")
        return (self.left_ee_apply_force + 
                self.right_ee_apply_force) / 2
                