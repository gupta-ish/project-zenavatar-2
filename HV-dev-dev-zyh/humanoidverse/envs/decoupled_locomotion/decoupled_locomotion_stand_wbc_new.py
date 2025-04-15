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
from isaac_utils.rotations import get_euler_xyz_in_tensor

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

from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_new import LeggedRobotDecoupledLocomotionStance

from loguru import logger

from scipy.stats import vonmises

from humanoidverse.utils.arm_ik import arm_ik

DEBUG = False
class LeggedRobotDecoupledLocomotionStanceWBC(LeggedRobotDecoupledLocomotionStance):
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

        # Upper body dof pos tracking termination buf
        self.far_upper_dof_pos_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    
    def step(self, actor_state):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = actor_state["actions"]

        self._pre_physics_step(actions)
        self._physics_step()
        self._post_physics_step()

        # if self.episode_length_buf[0] == 1:
        #     import ipdb; ipdb.set_trace()

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.config.robot.control.action_scale
        actions_lower_body = actions_scaled[:, :self.config.robot.lower_body_actions_dim]
        # TODO: Hardcode
        actions_residual_upper_body = actions_scaled[:, -self.config.robot.upper_body_actions_dim:]
        # print("actions_residual_upper_body: ", actions_residual_upper_body)
        actions_upper_body = actions_residual_upper_body # + self.ref_upper_dof_pos
        actions = torch.cat((actions_lower_body, actions_upper_body), dim=1)
        control_type = self.config.robot.control.control_type
        if control_type=="P":
            torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains*(self.simulator.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        
        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        else:
            return torques
    
    def _pre_compute_observations_callback(self, debug=DEBUG):
        # prepare quantities
        self.base_quat[:] = self.simulator.base_quat[:]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 7:10])
        # print("self.base_lin_vel", self.base_lin_vel)
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        # print("self.base_ang_vel", self.base_ang_vel)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # TODO [Yuanhang]: hardcode for Tairan's need :) should be removed
        if self.config.rewards.fix_upper_body:
            self.ref_upper_dof_pos *= 0.0
            # self.ref_upper_dof_pos[:, 4] = 0.3
            # self.ref_upper_dof_pos[:, 11] = -0.3
            # self.ref_upper_dof_pos[:, 6] = 1.
            # self.ref_upper_dof_pos[:, 13] = 1.
            return
        # Get the reference upper body joint positions
        offset = self.env_origins
        # print("env_ids_stance: ", env_ids_stance)
        # print("episode_length_buf: ", self.episode_length_buf)
        self.motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
        # print("motion_times: ", self.motion_times)
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=offset)
        
        # Update the upper body joint positions from motion library
        ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        self.ref_upper_dof_pos = ref_joint_pos[:, -self.config.robot.upper_body_actions_dim:] # [num_envs, upper_body_actions_dim]
        # Yuanhang: set waist's yaw, roll and pitch [12/21/2024]
        if self.fix_waist_yaw:
            # self.ref_upper_dof_pos[:, 0] *= 0.0
            self.ref_upper_dof_pos[:, 0] = self.fixed_waist_yaw_pos
        if self.fix_waist_roll:
            # self.ref_upper_dof_pos[:, 1] *= 0.0
            self.ref_upper_dof_pos[:, 1] = self.fixed_waist_roll_pos
        if self.fix_waist_pitch:
            # self.ref_upper_dof_pos[:, 2] *= 0.0
            self.ref_upper_dof_pos[:, 2] = self.fixed_waist_pitch_pos 
        if self.apply_waist_roll_only_when_stance:
            self.ref_upper_dof_pos[:, 1] *= (1 - self.commands[:, 4]) # only apply when stance
        if self.apply_waist_pitch_only_when_stance:
            self.ref_upper_dof_pos[:, 2] *= (1 - self.commands[:, 4]) # only apply when stance
        if self.apply_waist_yaw_only_when_stance:
            self.ref_upper_dof_pos[:, 0] *= (1 - self.commands[:, 4]) # only apply when stance
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
        if self.fix_waist_yaw:
            self.fixed_waist_yaw_pos[env_ids] = torch_rand_float(self.fix_waist_yaw_range[0], self.fix_waist_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze(1) * self.zero_fix_waist_yaw[env_ids]
        if self.fix_waist_pitch:
            self.fixed_waist_pitch_pos[env_ids] = torch_rand_float(self.fix_waist_pitch_range[0], self.fix_waist_pitch_range[1], (len(env_ids), 1), device=self.device).squeeze(1) * self.zero_fix_waist_pitch[env_ids]
        if self.fix_waist_roll:
            self.fixed_waist_roll_pos[env_ids] = torch_rand_float(self.fix_waist_roll_range[0], self.fix_waist_roll_range[1], (len(env_ids), 1), device=self.device).squeeze(1) * self.zero_fix_waist_roll[env_ids]
        
    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        
        self.commands = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
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
    
    def _update_reward_penalty_curriculum(self):
        """
        Update the penalty curriculum based on the average episode length.

        If the average episode length is below the penalty level down threshold,
        decrease the penalty scale by a certain level degree.
        If the average episode length is above the penalty level up threshold,
        increase the penalty scale by a certain level degree.
        Clip the penalty scale within the specified range.

        Returns:
            None
        """
        if self.average_episode_length < self.config.rewards.reward_penalty_level_down_threshold:
            self.reward_penalty_scale *= (1 - self.config.rewards.reward_penalty_degree)
        elif self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
            self.reward_penalty_scale *= (1 + self.config.rewards.reward_penalty_degree)

        self.reward_penalty_scale = np.clip(self.reward_penalty_scale, self.config.rewards.reward_min_penalty_scale, self.config.rewards.reward_max_penalty_scale)

    def _update_tasks_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # Push the robots randomly
        if self.config.domain_rand.push_robots:
            push_robot_env_ids = (self.push_robot_counter == (self.push_interval_s / self.dt).int()).nonzero(as_tuple=False).flatten()
            self.push_robot_counter[push_robot_env_ids] = 0
            self.push_robot_plot_counter[push_robot_env_ids] = 0
            self.push_interval_s[push_robot_env_ids] = torch.randint(self.config.domain_rand.push_interval_s[0], self.config.domain_rand.push_interval_s[1], (len(push_robot_env_ids),), device=self.device, requires_grad=False)
            self._push_robots(push_robot_env_ids)
        # Update locomotion commands
        if not self.is_evaluating:
            env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(self.commands[:, 3] - heading), 
            self.command_ranges["ang_vel_yaw"][0], 
            self.command_ranges["ang_vel_yaw"][1]
        )
        # only apply the velocity command if it is tapping command or not tapping in place
        # print("commands: ", self.commands)
        # print("tapping_in_place: ", self.tapping_in_place)
        # import pdb; pdb.set_trace()
        self.commands[:, 0] *= (self.commands[:, 4] * self.tapping_in_place[:, 0])
        self.commands[:, 1] *= (self.commands[:, 4] * self.tapping_in_place[:, 0])
        self.commands[:, 2] *= (self.commands[:, 4] * self.tapping_in_place[:, 0])
        # print("commands: ", self.commands)
        # If fixed, no need to update the upper body motion
        if self.config.rewards.fix_upper_body:
            return
        # Resample/Update upper body motions
        if self.config.resample_motion_when_training:
            if self.common_step_counter % self.resample_time_interval == 0:
                logger.info(f"Resampling motion at step {self.common_step_counter}")
                self.resample_motion()
        self.motion_len = self._motion_lib.get_motion_length(self.motion_ids)
        # motion_times = (self.episode_motion_length) * self.dt + self.motion_start_times # current frame
        # if self.config.robot.motion.reverse_motion:
        #     # Yichao: Here we consider a full video contains forward then its reverse motions, so double the length
        #     # reverse motions are addressed at get_motion_state in motion_lib
        #     env_ids = torch.where(motion_times > 2 * self.motion_len)[0] # check if the motion is finished
        # else: 
        #     env_ids = torch.where(motion_times > self.motion_len)[0] # check if the motion is finished
        #     self._resample_motion_times(env_ids) # Yuanhang: resample the motion start times only when non-reverse motion
        # self.episode_motion_length[env_ids] = 0 # reset the episode motion length

    def _check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = 0
        # self.time_out_buf = 0
        # Note: DO NOT USE FOLLOWING TWO LINES STYLE
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0

        self._update_reset_buf()
        self._update_timeout_buf()
        self._update_far_upper_dof_pos_buf()
        # print("reset_buf: ", self.reset_buf)
        # print("time_out_buf: ", self.time_out_buf)
        # print("far_upper_dof_pos_buf: ", self.far_upper_dof_pos_buf)

    def _update_timeout_buf(self):
        super()._update_timeout_buf()
        if self.config.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.time_out_buf |= current_time > self.motion_len
        # import ipdb; ipdb.set_trace()
        # print("time_out_buf: ", self.time_out_buf)
        self.reset_buf |= self.time_out_buf
    
    def _update_far_upper_dof_pos_buf(self):
        if self.config.termination.terminate_when_low_upper_dof_tracking:
            # Yuanhang: upper body dof position tracking error
            # upper_body_dof_pos_tracking_reward = self._reward_tracking_upper_body_dofs()
            # self.far_upper_dof_pos_buf[:] = upper_body_dof_pos_tracking_reward < self.config.termination_scales.terminate_when_low_upper_dof_tracking_threshold
            # ExBody
            dof_dev = torch.exp(-0.5 * torch.norm((self.simulator.dof_pos[:, self.upper_dof_indices] - self.ref_upper_dof_pos), dim=1))
            self.far_upper_dof_pos_buf[:] = dof_dev < self.config.termination_scales.terminate_when_low_upper_dof_tracking_threshold
            # print("upper_body_dof_pos_tracking_error: ", upper_body_dof_pos_tracking_error)
            self.reset_buf |= self.far_upper_dof_pos_buf
        else:
            self.far_upper_dof_pos_buf[:] = False

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
    
    ########################### FEET REWARDS ###########################
    
    ######################## LIMITS REWARDS #########################
    def _reward_limits_dof_pos(self):
        # Penalize dof positions too close to the limit (lower body only)
        out_of_limits = -(self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_limits_dof_vel(self):
        # Penalize dof velocities too close to the limit (lower body only)
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.simulator.dof_vel) - self.simulator.dof_vel_limits[:, 1]).clip(min=0., max=1.), dim=1)

    def _reward_limits_torque(self):
        # penalize torques too close to the limit (lower body only)
        return torch.sum((torch.abs(self.torques) - self.torque_limits[:, 1]).clip(min=0.), dim=1)
    
    ######################### PENALTY REWARDS #########################
    def _reward_penalty_torques(self):
        # Penalize torques (lower body only)
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_penalty_dof_vel(self):
        # Penalize dof velocities (lower body only)
        return torch.sum(torch.square(self.simulator.dof_vel), dim=1)
    
    def _reward_penalty_dof_acc(self):
        # Penalize dof accelerations (lower body only)
        return torch.sum(torch.square((self.last_dof_vel - self.simulator.dof_vel) / self.dt), dim=1)
    
    def _reward_penalty_action_rate(self):
        # Penalize changes in actions (lower body only)
        lower_body_action_rate = torch.sum(torch.square(self.last_actions[:, :self.config.robot.lower_body_actions_dim] - self.actions[:, :self.config.robot.lower_body_actions_dim]), dim=1)
        upper_body_action_rate = torch.sum(torch.square(self.last_actions[:, -self.config.robot.upper_body_actions_dim:] - self.actions[:, -self.config.robot.upper_body_actions_dim:]), dim=1)
        return lower_body_action_rate + upper_body_action_rate # Yuanhang: lower body action rate is more important
    
    def _reward_penalty_waist_dofs(self):
        # Penalize waist dof positions (yaw, pitch, roll)
        waist_pos = self.simulator.dof_pos[:, self.waist_dof_indices]
        waist_dofs_error = torch.sum(torch.square(waist_pos - self.ref_upper_dof_pos[:, 0:3]), dim=1)
        return waist_dofs_error
    
    def _reward_tracking_waist_dofs(self):
        # Reward the difference between the waist dof pos and the reference
        waist_pos = self.simulator.dof_pos[:, self.waist_dof_indices]
        waist_dofs_error =  torch.sum(torch.square(waist_pos - self.ref_upper_dof_pos[:, 0:3]), dim=1)
        return torch.exp(-waist_dofs_error/self.config.rewards.reward_tracking_sigma.waist_dofs)
    
    def _reward_tracking_arm_dofs(self):
        # Reward the difference between the arm dof pos and the reference
        arm_pos = self.simulator.dof_pos[:, self.arm_dof_indices]
        arm_dofs_error =  torch.sum(torch.square(arm_pos - self.ref_upper_dof_pos[:, 3:]), dim=1)
        return torch.exp(-arm_dofs_error/self.config.rewards.reward_tracking_sigma.arm_dofs)
    
    def _reward_tracking_upper_body_dofs(self):
        # Reward the difference between the waist dof pos and the reference
        upper_body_pos = self.simulator.dof_pos[:, self.upper_dof_indices]
        upper_body_dofs_error =  torch.sum(torch.square(upper_body_pos - self.ref_upper_dof_pos), dim=1)
        return torch.exp(-upper_body_dofs_error/self.config.rewards.reward_tracking_sigma.upper_body_dofs)

    def _reward_penalty_upper_body_dofs_freeze(self):
        # returns keep the upper body joint angles close to the default
        assert self.config.robot.has_upper_body_dof
        # import ipdb; ipdb.set_trace()
        deviation = torch.abs(self.simulator.dof_pos[:, self.upper_dof_indices] - self.default_dof_pos[:, self.upper_dof_indices])
        # print(torch.sum(deviation, dim=1))
        return torch.sum(deviation, dim=1)
    
    def _get_obs_command_lin_vel(self):
        # print("commands: ", self.commands)
        return self.commands[:, :2]
    
    def _get_obs_command_ang_vel(self):
        return self.commands[:, 2:3]

    ######################### Observations #########################
    def _get_obs_base_orientation(self):
        return self.base_quat[:, 0:4]
    
    def _get_obs_actions(self):
        return self.actions