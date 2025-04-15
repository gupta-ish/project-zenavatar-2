import numpy as np
import time
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
import threading
# from pynput import keyboard
import argparse
import yaml
import sys
sys.path.append('./rl_policy')

from base_policy import BasePolicy
from sim2real.utils.key_cmd import KeyboardPolicy

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

class LocomotionPolicy(BasePolicy):
    def __init__(self, 
                 config, 
                 node, 
                 model_path, 
                 use_jit,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4,
                 use_mocap=False):
        super().__init__(config, 
                         node, 
                         model_path, 
                         use_jit,
                         rl_rate, 
                         policy_action_scale, 
                         decimation)
        self.lin_vel_command = np.array([[0., 0.]])
        self.ang_vel_command = np.array([[0.]])

    def prepare_obs_for_rl(self, robot_state_data):
        # robot_state_data [:3]: robot base pos
        # robot_state_data [3:7]: robot base quaternion
        # robot_state_data [7:7+dof_num]: joint angles 
        # robot_state_data [7+dof_num: 7+dof_num+3]: base linear velocity
        # robot_state_data [7+dof_num+3: 7+dof_num+6]: base angular velocity
        # robot_state_data [7+dof_num+6: 7+dof_num+6+dof_num]: joint velocities
        
        # RL observation preparation
        base_quat = robot_state_data[:, 3:7]
        base_ang_vel = robot_state_data[:, 7+self.num_dofs+3:7+self.num_dofs+6]
        dof_pos = robot_state_data[:, 7:7+self.num_dofs]
        dof_vel = robot_state_data[:, 7+self.num_dofs+6:7+self.num_dofs+6+self.num_dofs]

        dof_pos_minus_default = dof_pos - self.default_dof_angles

        v = np.array([[0, 0, -1]])

        projected_gravity = quat_rotate_inverse_numpy(base_quat, v)
        
        if self.use_mocap:
            base_lin_vel = self.mocap_lin_vel.reshape(1, -1)
            base_pos = self.mocap_pos.reshape(1, -1)
            base_quat = self.mocap_quat.reshape(1, -1)
            obs = np.concatenate([self.last_policy_action[:, :12], 
                                    base_ang_vel*0.25, 
                                    base_lin_vel*2.0,
                                    base_pos,
                                    base_quat,
                                    self.ang_vel_command, 
                                    self.lin_vel_command, 
                                    dof_pos_minus_default[:, :12], 
                                    dof_vel[:, :12]*0.05,
                                    projected_gravity
                                    ], axis=1)
        
        else:
            obs = np.concatenate([self.last_policy_action, 
                                    base_ang_vel*0.25, 
                                    self.ang_vel_command, 
                                    self.lin_vel_command, 
                                    dof_pos_minus_default, 
                                    dof_vel*0.05,
                                    projected_gravity
                                    ], axis=1)

        # examine obs
        # print("last_policy_action", self.last_policy_action)
        # print("base_ang_vel", base_ang_vel)
        # print("ang_vel_command", self.ang_vel_command)
        # print("lin_vel_command", self.lin_vel_command)
        # print("dof_pos_minus_default", dof_pos_minus_default)
        # print("dof_vel", dof_vel)
        # print("projected_gravity", projected_gravity)
        return obs.astype(np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--use_jit', action='store_true', default=False, help='use jit')
    parser.add_argument('--use_mocap', action='store_true', default=False, help='use mocap')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    policy = LocomotionPolicy(config=config, 
                              model_path=args.model_path, 
                              use_jit=args.use_jit,
                              rl_rate=50, 
                              decimation=4)
    policy.run()