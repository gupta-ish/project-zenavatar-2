import numpy as np
import time
from scipy.spatial.transform import Rotation
import threading
import argparse
import yaml
import sys
sys.path.append('./rl_policy')

from base_policy import BasePolicy
# from pynput import keyboard
from sshkeyboard import listen_keyboard
from termcolor import colored

from utils.robot_arm_ik_reduce_wrist import G1_29_ArmIK
from utils.EE_motion_planner import EE_MotionPlanner
import pinocchio as pin

def quat_rotate_inverse_numpy(q, v):
    shape = q.shape
    # q_w corresponds to the scalar part of the quaternion
    q_w = q[:, 0]
    # q_vec corresponds to the vector part of the quaternion
    q_vec = q[:, 1:]

    # Calculate a
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]

    # Calculate b
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0

    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return a - b + c

def clock_input():
    t = time.time()
    t -= int(t / 1000) * 1000

    frequency = 1.5
    phase = 0.5

    gait_indices = t * frequency - int(t * frequency)
    foot_indices = np.array([phase, 0]) + gait_indices

    clock_inputs = np.sin(2 * np.pi * foot_indices)

    return clock_inputs[None, :]

class WBCLocomotionStandPolicy(BasePolicy):
    def __init__(self, 
                 config, 
                 model_path, 
                 use_jit,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4,
                 use_mocap=False):
        super().__init__(config, 
                         model_path, 
                         use_jit,
                         rl_rate, 
                         policy_action_scale, 
                         decimation)
        self.use_clock_input = False

        self.num_upper_dofs = self.config['NUM_UPPER_BODY_JOINTS']
        self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
        self.ref_upper_dof_pos *= 0.0
        self.ref_upper_dof_pos[:, 1] = 0.3
        self.ref_upper_dof_pos[:, 8] = -0.3
        self.ref_upper_dof_pos[:, 3] = 0.8
        self.ref_upper_dof_pos[:, 10] = 0.8
        self.base_height_command = np.array([[0.75]])
        # waist pitch offset
        # self.ref_upper_dof_pos[:, 2] = 0.4
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.gait_period = self.config["GAIT_PERIOD"]
        self.phase_time = np.zeros((1, 1))

        self.obs_list = self.config['obs_list']
        self.obs_dims = self.config['obs_dims']
        self.obs_dim = self._calculate_obs_dim()
        self.history_length = self.config["history_length"]
        self.obs_buf = np.zeros((1, self.obs_dim * self.history_length))
        
        self.residual_upper_body_action = self.config.get("residual_upper_body_action", False)

        self.waist_dofs_command = np.zeros((1, 3))
        self.init_upper_body_controller()
        self.init_motion_planner()

    def init_upper_body_controller(self):
        self.upper_body_controller = G1_29_ArmIK(Unit_Test=False, Visualization=False)
        self.waypoint_index = 0
        self.speed_factor = 0.05
        self.base_z_offset = 0.8
        # Initialize waypoints
        self.degrees = 0
        self.theta = np.radians(self.degrees)
        self.EE_left_R = np.array([[np.cos(-self.theta), -np.sin(-self.theta), 0],
                                   [np.sin(-self.theta),  np.cos(-self.theta), 0],
                                   [                 0,                   0, 1]])
        self.EE_right_R = np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
                                   [np.sin(self.theta),  np.cos(self.theta), 0],
                                   [                 0,                   0, 1]])
        self.EE_left_x = 0.30
        self.EE_right_x = 0.30
        self.EE_left_y = 0.13
        self.EE_right_y = -0.13
        self.EE_left_z = 0.08
        self.EE_right_z = 0.08
        self.update_waypoints()
        # Initialize external force
        self.EE_efrc_L=np.array([0, 0, 0, 0, 0, 0])
        self.EE_efrc_R=np.array([0, 0, 0, 0, 0, 0])
        # Initialize interpolated positions and orientations
        self.upper_body_controller.set_initial_poses(
            self.waypoints_left[0].translation,
            self.waypoints_right[0].translation,
            self.waypoints_left[0].rotation,
            self.waypoints_right[0].rotation
        )
    
    def init_motion_planner(self):
        self.MP = EE_MotionPlanner()
        self.MP_enabled = False
        self.MP_waypoints_left = [pin.SE3(np.eye(3), np.array([0.25, 0.18 , 0.08])),
                                  pin.SE3(np.eye(3), np.array([0.35, 0.18 , 0.08])),
                                  pin.SE3(np.eye(3), np.array([0.35, 0.18 , 0.2])), ]
        self.MP_waypoints_right = [pin.SE3(np.eye(3), np.array([0.25, -0.18 , 0.08])),
                                   pin.SE3(np.eye(3), np.array([0.35, -0.18 , 0.08])), 
                                   pin.SE3(np.eye(3), np.array([0.35, -0.18 , 0.2])), ]
       

    def _calculate_obs_dim(self):
        obs_dim = 0
        for key in self.obs_list:
            obs_dim += self.obs_dims[key]
        return obs_dim

    def _get_obs_phase_time(self):
        cur_time = time.perf_counter() * self.stand_command[0, 0]
        # print("cur_time: ", cur_time)
        phase_time = cur_time % self.gait_period / self.gait_period
        # print("phase_time: ", phase_time)
        self.phase_time[:, 0] = phase_time
        return self.phase_time

    def prepare_obs_for_rl(self, robot_state_data):
        # robot_state [:2]: timestamps
        # robot_state [2:5]: robot base pos
        # robot_state [5:9]: robot base orientation
        # robot_state [9:9+dof_num]: joint angles 
        # robot_state [9+dof_num: 9+dof_num+3]: base linear velocity
        # robot_state [9+dof_num+3: 9+dof_num+6]: base angular velocity
        # robot_state [9+dof_num+6: 9+dof_num+6+dof_num]: joint velocities
        # RL observation preparation
        base_quat = robot_state_data[:, 3:7]
        base_ang_vel = robot_state_data[:, 7+self.num_dofs+3:7+self.num_dofs+6]
        dof_pos = robot_state_data[:, 7:7+self.num_dofs]
        dof_vel = robot_state_data[:, 7+self.num_dofs+6:7+self.num_dofs+6+self.num_dofs]


        dof_pos_minus_default = dof_pos - self.default_dof_angles

        v = np.array([[0, 0, -1]])

        projected_gravity = quat_rotate_inverse_numpy(base_quat, v)
        
        phase_time = self._get_obs_phase_time()
        sin_phase = np.sin(2*np.pi*phase_time)
        cos_phase = np.cos(2*np.pi*phase_time)
        
        # import ipdb; ipdb.set_trace()   
        # print(base_ang_vel)
        """
        base_ang_vel,
        projected_gravity,
        command_lin_vel,
        command_ang_vel,
        command_stand,
        command_base_height,
        command_waist_dofs,
        ref_upper_dof_pos,
        dof_pos,
        dof_vel,
        actions,

        sorted order:
        actions,
        base_ang_vel,
        command_ang_vel,
        command_base_height,
        command_lin_vel,
        command_stand,
        command_waist_dofs,
        dof_pos,
        dof_vel,
        projected_gravity,
        ref_upper_dof_pos,
        """
        # curr_obs = np.concatenate([base_ang_vel*0.25, 
        #                            projected_gravity,
        #                            self.lin_vel_command,
        #                            self.ang_vel_command, 
        #                            self.stand_command,
        #                            self.base_height_command*2.0,
        #                            self.waist_dofs_command,
        #                            self.ref_upper_dof_pos,
        #                            dof_pos_minus_default, 
        #                            dof_vel*0.05,
        #                            self.last_policy_action, 
        #                            ], axis=1)
        curr_obs = np.concatenate([self.last_policy_action,
                                   base_ang_vel*0.25,
                                   self.ang_vel_command,
                                   self.base_height_command*2.0,
                                   self.lin_vel_command,
                                   self.stand_command,
                                   self.waist_dofs_command,
                                   dof_pos_minus_default,
                                   dof_vel*0.05,
                                   projected_gravity,
                                   self.ref_upper_dof_pos,
                                   ], axis=1)

        self.obs_buf = np.concatenate((self.obs_buf[:, self.obs_dim:(self.obs_dim*self.history_length)], curr_obs), axis=1)
        obs = self.obs_buf.copy()
        # import ipdb; ipdb.set_trace()
        # examine obs
        # print("last_policy_action", self.last_policy_action)
        # print("base_ang_vel", base_ang_vel)
        # print("ang_vel_command", self.ang_vel_command)
        # print("lin_vel_command", self.lin_vel_command)
        # print("stand_command", self.stand_command)
        # print("dof_pos_minus_default", dof_pos_minus_default)
        # print("dof_vel", dof_vel)
        # print("projected_gravity", projected_gravity)
        # print("ref_upper_dof_pos", self.ref_upper_dof_pos)
        return obs.astype(np.float32)

    def rl_inference(self, robot_state_data):
        obs = self.prepare_obs_for_rl(robot_state_data)
        # import ipdb; ipdb.set_trace()
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)
        
        # WBC actions
        self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale

        # Lower body actions
        lower_body_action = scaled_policy_action[:, :(self.num_dofs-self.num_upper_dofs)]
        # Upper body actions
        upper_body_action = scaled_policy_action[:, -self.num_upper_dofs:]
        if self.residual_upper_body_action:
            scaled_policy_action = np.concatenate([lower_body_action, self.ref_upper_dof_pos + upper_body_action], axis=1)
        else:
            scaled_policy_action = np.concatenate([lower_body_action, upper_body_action], axis=1)

        return scaled_policy_action

    def policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        # Get states
        self.robot_state_data = self.state_processor._prepare_low_state()
        self.robot_state_data_shm[0] = self.robot_state_data
        # Apply upper body controller
        if self.upper_body_controller:
           # Get whole body qpos, qvel, qtau
           qpos = self.robot_state_data[:, 7:7+self.num_dofs]
           qvel = self.robot_state_data[:, 7+self.num_dofs+6:7+self.num_dofs+6+self.num_dofs]
           qtau = self.robot_state_data[:, 7+self.num_dofs+6+self.num_dofs:7+self.num_dofs+6+self.num_dofs+self.num_dofs]
           if self.MP_enabled == False:
               # Control upper qpos
                upper_body_qpos, upper_body_tauff = self.upper_body_controller.get_q_tau(
                    self.waypoints_left[0],
                    self.waypoints_right[0],
                    self.EE_efrc_L, self.EE_efrc_R,
                )
                # # Control upper tau
                arm_reduced_joint_indices = [0, 1, 2, 3, 7, 8, 9, 10]
                for i, idx in enumerate(arm_reduced_joint_indices):
                    self.ref_upper_dof_pos[0, idx] = upper_body_qpos[i]
                    cmd_tau[idx] = upper_body_tauff[i]
                # Zero out wrist joints
                wrist_joint_indices = [ 19, 20, 21, 26, 27, 28]
                for idx in wrist_joint_indices:
                    cmd_q[idx] = 0.0
                    cmd_dq[idx] = 0.0
                    cmd_tau[idx] = 0.0
           else:
                # Control upper qpos
                upper_body_qpos, upper_body_tauff = self.upper_body_controller.get_q_tau(
                    self.MP_waypoints_left[self.waypoint_index],
                    self.MP_waypoints_right[self.waypoint_index],
                    self.EE_efrc_L, self.EE_efrc_R,
                )
                self.ref_upper_dof_pos[:, -14:] = upper_body_qpos
                # Control upper tau
                cmd_tau[-14:] = upper_body_tauff

                EE_error = self.upper_body_controller.get_target_waypoint_error(
                    qpos[:, -14:],
                    self.MP_waypoints_left[self.waypoint_index],
                    self.MP_waypoints_right[self.waypoint_index])
                if EE_error < 0.01:
                        self.waypoint_index += 1
                        if self.waypoint_index >= len(self.MP_waypoints_left):
                            self.waypoint_index = len(self.MP_waypoints_left) - 1
        
        # Get policy action
        scaled_policy_action = self.rl_inference(self.robot_state_data)
        if self.get_ready_state:
            # 1. Set to Default Joint Position: interpolate from current dof_pos to default angles
            q_target = self.get_init_target(self.robot_state_data)
            if self.init_count > 500:
                self.init_count = 500
        elif not self.use_policy_action:
            # 2. No Policy Action: set to zero
            q_target = self.robot_state_data[:, 7:7+self.num_dofs]
        else:
            # 3. Policy Action: apply policy action to current joint angles
            q_target = scaled_policy_action + self.default_dof_angles
        # import ipdb; ipdb.set_trace()
        # Clip q target
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

        # Send command
        cmd_q = q_target[0]
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)

    
    def update_waypoints(self):
        self.theta = np.radians(self.degrees)
        self.EE_left_R = np.array([[np.cos(-self.theta), -np.sin(-self.theta), 0],
                                   [np.sin(-self.theta),  np.cos(-self.theta), 0],
                                   [                 0,                   0, 1]])
        self.EE_right_R = np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
                                   [np.sin(self.theta),  np.cos(self.theta), 0],
                                   [                 0,                   0, 1]])
        self.waypoints_left = [
            pin.SE3(self.EE_left_R.astype(np.float64), np.array([self.EE_left_x, self.EE_left_y , self.EE_left_z]))]
        self.waypoints_right = [
            pin.SE3(self.EE_right_R.astype(np.float64), np.array([self.EE_right_x, self.EE_right_y , self.EE_right_z]))]
        
    
    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        if keycode == ",":
            self.waist_dofs_command[:, 0] -= 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif keycode == ".":
            self.waist_dofs_command[:, 0] += 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
    
    def handle_joystick_button(self, cur_key):
        super().handle_joystick_button(cur_key)
        if cur_key == "Y+up":
           self.ref_upper_dof_pos[:, 2] -= 0.05
           self.logger.info(colored(f"waist pitch: {self.ref_upper_dof_pos[:, 2]}", "green"))
        elif cur_key == "Y+down":
           self.ref_upper_dof_pos[:, 2] += 0.05
           self.logger.info(colored(f"waist pitch: {self.ref_upper_dof_pos[:, 2]}", "green"))
        elif cur_key == "R1+up":
            self.EE_left_x += 0.05
            self.EE_right_x += 0.05
            self.update_waypoints()
            self.logger.info(colored(f"EE X command: {self.EE_left_x}", "green"))
        elif cur_key == "R1+down":
            self.EE_left_x -= 0.05
            self.EE_right_x -= 0.05
            self.update_waypoints()
            self.logger.info(colored(f"EE X command: {self.EE_left_x}", "green"))
        elif cur_key == "R1+left":
            self.EE_left_y += 0.02
            self.EE_right_y -= 0.02
            self.update_waypoints()
            self.logger.info(colored(f"EE Y command: {self.EE_left_y}", "green"))
        elif cur_key == "R1+right":
            self.EE_left_y -= 0.02 
            self.EE_right_y += 0.02
            self.update_waypoints()
            self.logger.info(colored(f"EE Y command: {self.EE_left_y}", "green"))
        elif cur_key == "X+up":
            self.EE_left_z += 0.05
            self.EE_right_z += 0.05
            self.update_waypoints()
            self.logger.info(colored(f"EE Z command: {self.EE_left_z}", "green"))
        elif cur_key == "X+down":
            self.EE_left_z -= 0.05
            self.EE_right_z -= 0.05
            self.update_waypoints()
            self.logger.info(colored(f"EE Z command: {self.EE_left_z}", "green"))
        elif cur_key == "X+left":
            self.degrees -= 5
            self.update_waypoints()
            self.logger.info(colored(f"EE Wrist Yaw: {self.degrees}", "green"))
        elif cur_key == "X+right":
            self.degrees += 5
            self.update_waypoints()
            self.logger.info(colored(f"EE Wrist Yaw: {self.degrees}", "green"))
        elif cur_key == "select+left":
            self.waist_dofs_command[:, 0] -= 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif cur_key == "select+right":
            self.waist_dofs_command[:, 0] += 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--use_jit', action='store_true', default=False, help='use jit')
    parser.add_argument('--use_mocap', action='store_true', default=False, help='use mocap')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    policy = WBCLocomotionStandPolicy(config=config, 
                                      model_path=args.model_path, 
                                      use_jit=args.use_jit,
                                      rl_rate=50, 
                                      decimation=4)
    policy.run()