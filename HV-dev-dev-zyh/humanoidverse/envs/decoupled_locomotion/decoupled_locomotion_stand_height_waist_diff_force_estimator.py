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
from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_height_waist_diff_force import LeggedRobotDecoupledLocomotionStanceHeightWaistForce
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

from humanoidverse.utils.common import unnormalize, normalize

from humanoidverse.utils.torch_utils import quat_from_euler_xyz, quat_rotate_inverse

DEBUG = False
class LeggedRobotDecoupledLocomotionStanceHeightWaistForceEstimator(LeggedRobotDecoupledLocomotionStanceHeightWaistForce):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.left_ee_estimated_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_estimated_force = torch.zeros((self.num_envs, 3), device=self.device)

        self.left_ee_comp = torch.zeros((self.num_envs, self.num_upper_dofs), device=self.device)
        self.right_ee_comp = torch.zeros((self.num_envs, self.num_upper_dofs), device=self.device)
        
        self.apply_force_estimation_error_threshold = self.config.get("apply_force_estimation_error_threshold", 5.0)
        self.estimated_env_ids = torch.where(self.commands[:, 4] == 0)[0]
        self.apply_or_estimated_force = torch.zeros((self.num_envs, 1), device=self.device)
        
    def _init_buffers(self):
        super()._init_buffers()
    
    def _update_tasks_callback(self):
        super()._update_tasks_callback()
    
    def step(self, actor_state):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = actor_state["actions"]

        if "left_ee_force_estimator_output" in actor_state:
            # Note: The estimator output is scaled, so we need to unscale it here
            self.left_ee_estimated_force = unnormalize(actor_state["left_ee_force_estimator_output"],
                                                       self.force_range_low,
                                                       self.force_range_high)
        
        if "right_ee_force_estimator_output" in actor_state:
            # Note: The estimator output is scaled, so we need to unscale it here
            self.right_ee_estimated_force = unnormalize(actor_state["right_ee_force_estimator_output"],
                                                        self.force_range_low,
                                                        self.force_range_high)

        self._pre_physics_step(actions)
        self._physics_step()
        self._post_physics_step()

        # if self.episode_length_buf[0] == 1:
        #     import ipdb; ipdb.set_trace()

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras
    
    def _apply_force_in_physics_step(self):
        # Apply the force in the physics step
        self.torques = self._compute_torques(self.actions_after_delay).view(self.torques.shape)
        
        left_hand_force, right_hand_force = self._calculate_max_ee_force()
        
        if self.is_evaluating:
            # left_hand_force = torch.zeros_like(left_hand_force)
            # right_hand_force = torch.zeros_like(right_hand_force)
            # left_hand_force[:, 1] = 10.0
            # right_hand_force[:, 1] = 10.0
            # left_hand_force[:, 2] = -40.0
            # right_hand_force[:, 2] = -40.0
            pass
        
        # Rotate the force from global to base frame
        self.left_ee_apply_force = quat_rotate_inverse(self.base_quat, left_hand_force)
        self.right_ee_apply_force = quat_rotate_inverse(self.base_quat, right_hand_force)

        # Apply the force to the hand links
        self.apply_force_tensor[:, self.left_hand_link_index, :] = left_hand_force
        self.apply_force_tensor[:, self.right_hand_link_index, :] = right_hand_force
        
        if self.config.apply_force_in_physics_step:
            self.simulator.apply_rigid_body_force_at_pos_tensor(self.apply_force_tensor, self.apply_force_pos_tensor)
        
        # TODO [Yuanhang]: Pick the stance envs based on the estimated force error
        global_left_ee_estimated_force = quat_rotate(self.base_quat, self.left_ee_estimated_force)
        global_right_ee_estimated_force = quat_rotate(self.base_quat, self.right_ee_estimated_force)
        global_left_ee_estimation_error = torch.norm(left_hand_force - global_left_ee_estimated_force, dim=1)
        global_right_ee_estimation_error = torch.norm(right_hand_force - global_right_ee_estimated_force, dim=1)
        if self.config.update_estimator_only_when_stance:
            self.estimated_env_ids = torch.where((self.apply_force_scale[:, 0] > 0.25) # Hardcoded threshold
                                                    & (self.commands[:, 4] == 0)
                                                    # & (global_left_ee_estimation_error < self.apply_force_estimation_error_threshold) 
                                                    # & (global_right_ee_estimation_error < self.apply_force_estimation_error_threshold)
                                                    )[0]
        else:
            self.estimated_env_ids = torch.where((self.apply_force_scale[:, 0] > 0.25) # Hardcoded threshold
                                                    # & (self.commands[:, 4] == 0)
                                                    # & (global_left_ee_estimation_error < self.apply_force_estimation_error_threshold) 
                                                    # & (global_right_ee_estimation_error < self.apply_force_estimation_error_threshold)
                                                    )[0]
        left_hand_force[self.estimated_env_ids] = global_left_ee_estimated_force[self.estimated_env_ids]
        right_hand_force[self.estimated_env_ids] = global_right_ee_estimated_force[self.estimated_env_ids]
        if self.is_evaluating:
            # print("self.estimated_env_ids: ", self.estimated_env_ids)
            # print(f"Estimated Force Left: {global_left_ee_estimated_force}, Right: {global_right_ee_estimated_force}")
            # print(f"Estimated Force Left: {global_left_ee_estimation_error}, Right: {global_right_ee_estimation_error}")
            pass
        self.apply_or_estimated_force[:] = 0.0
        if self.config.update_estimator_only_when_stance:
            valid_estimated_env_ids = torch.where((self.commands[:, 4] == 0)
                                                        & (self.apply_force_scale[:, 0] > 0.25) # Hardcoded threshold
                                                        & (global_left_ee_estimation_error < self.apply_force_estimation_error_threshold) 
                                                        & (global_right_ee_estimation_error < self.apply_force_estimation_error_threshold)
                                                        )[0]
        else:
            valid_estimated_env_ids = torch.where((self.apply_force_scale[:, 0] > 0.25) # Hardcoded threshold
                                                        & (global_left_ee_estimation_error < self.apply_force_estimation_error_threshold) 
                                                        & (global_right_ee_estimation_error < self.apply_force_estimation_error_threshold)
                                                        )[0]
        self.apply_or_estimated_force[valid_estimated_env_ids] = 1.0
        
        if self.config.apply_force_compensation_in_physics_step:
            self._apply_force_compensation_in_physics_step(left_hand_force, right_hand_force)
        
        self.simulator.apply_torques_at_dof(self.torques)
    
    # def _update_timeout_buf(self):
    #     super()._update_timeout_buf()
    #     if self.config.termination.terminate_when_motion_end:
    #         current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
    #         self.time_out_buf |= current_time > self.motion_len
    #     # import ipdb; ipdb.set_trace()
    #     # print("time_out_buf: ", self.time_out_buf)
    #     self.reset_buf |= self.time_out_buf
    
    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        
    def _draw_debug_vis(self):
        super()._draw_debug_vis()

    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        if self.is_evaluating:
            # self.ref_upper_dof_pos *= 0.0
            # self.ref_upper_dof_pos[:, 1] = 0.3
            # self.ref_upper_dof_pos[:, 8] = -0.3
            # self.ref_upper_dof_pos[:, 2] = 0.8
            # self.ref_upper_dof_pos[:, 10] = 0.8
            # self.ref_upper_dof_pos[:, 4] = 0.3
            # self.ref_upper_dof_pos[:, 11] = -0.3
            # self.ref_upper_dof_pos[:, 6] = 0.8
            # self.ref_upper_dof_pos[:, 13] = 0.8
            pass
    ########################### FEET REWARDS ###########################
    
    ######################## ESTIMATION REWARDS ########################
    
    def _reward_force_estimation(self):
        estimated_envs_mask = torch.zeros((self.num_envs, ), device=self.device)
        estimated_envs_mask[self.estimated_env_ids] = 1.0
        left_ee_estimated_force_error = normalize(self.left_ee_estimated_force, self.force_range_low, self.force_range_high) - \
                                        normalize(self.left_ee_apply_force, self.force_range_low, self.force_range_high)
        right_ee_estimated_force_error = normalize(self.right_ee_estimated_force, self.force_range_low, self.force_range_high) - \
                                         normalize(self.right_ee_apply_force, self.force_range_low, self.force_range_high)
        estimated_error = torch.sum(torch.square(left_ee_estimated_force_error) + torch.square(right_ee_estimated_force_error), dim=1)
        return torch.exp(-estimated_error / self.config.rewards.reward_tracking_sigma.force_estimation) * (self.apply_force_scale[:, 0] > 0.3).float() * estimated_envs_mask
    
    ######################### Observations #########################
    def _get_obs_dof_forces(self):
        return self.simulator.dof_forces

    def _get_obs_left_ee_comp(self):
        return self.left_ee_comp
    
    def _get_obs_right_ee_comp(self):
        return self.right_ee_comp
    
    def _get_obs_left_ee_estimated_force(self):
        return self.left_ee_estimated_force
    
    def _get_obs_right_ee_estimated_force(self):
        return self.right_ee_estimated_force
    
    def _get_obs_apply_force_scale(self):
        return self.apply_force_scale

    def _get_obs_apply_or_estimated_force(self):
        return self.apply_or_estimated_force
    
    def _get_obs_history_apply_force(self,):
        assert "history_apply_force" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_apply_force']
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_estimated_force(self,):
        assert "history_estimated_force" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_estimated_force']
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_estimator(self,):
        assert "history_estimator" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_estimator']
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)