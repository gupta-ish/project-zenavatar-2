import numpy as np
import time
from scipy.spatial.transform import Rotation
import threading
import onnxruntime
# from pynput import keyboard
import argparse
import yaml
import sys
from scipy.spatial.transform import Rotation as R
sys.path.append('./rl_policy')
import cv2
from termcolor import colored
from decoupled_loco_manipulation_force import DecoupledLocomotionManipulationForcePolicy
from multiprocessing import shared_memory, Array, Lock
from sim2real.teleop.image_server.image_client import ImageClient
from sim2real.teleop.open_television.tv_wrapper import TeleVisionWrapper

from utils.robot_arm_ik import G1_29_ArmIK
import pinocchio as pin
from sim2real.utils.util import normalize, unnormalize
import pinocchio as pin

from sim2real.utils.util import quat_rotate_numpy, quat_rotate_inverse_numpy

def clock_input():
    t = time.time()
    t -= int(t / 1000) * 1000

    frequency = 1.5
    phase = 0.5

    gait_indices = t * frequency - int(t * frequency)
    foot_indices = np.array([phase, 0]) + gait_indices

    clock_inputs = np.sin(2 * np.pi * foot_indices)

    return clock_inputs[None, :]

class DecoupledLocomotionManipulationForcePolicy(DecoupledLocomotionManipulationForcePolicy):
    def __init__(self, 
                 config, 
                 avp_config,
                 model_path, 
                 use_jit,
                 rl_rate=50, 
                 policy_action_scale=0.25, 
                 decimation=4,
                 use_mocap=False):
        super().__init__(config, 
                         avp_config,
                         model_path, 
                         use_jit,
                         rl_rate, 
                         policy_action_scale, 
                         decimation,
                         use_mocap)
        
        self.waist_dofs_command = np.zeros((1, 3))
    
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
        
        # phase_time = self._get_obs_phase_time()
        # sin_phase = np.sin(2*np.pi*phase_time)
        # cos_phase = np.cos(2*np.pi*phase_time)

        # Yuanhang: get the estimated force here
        # if self.init_force_shm and self.estimating_force: 
        #     estimated_ee_force = self.estimated_ee_force_shm.copy()
        #     print("estimated_ee_force: ", estimated_ee_force)

        # import ipdb; ipdb.set_trace()   
        # print(base_ang_vel)
        if self.use_history:
            history_actor = self._get_obs_history_actor()
            history_actor *= self.obs_scales["history_actor"]
            history_estimator = self._get_obs_history_estimator()
            history_estimator *= self.obs_scales["history_estimator"]
            actor_obs = np.concatenate([self.last_policy_action, 
                                    base_ang_vel*0.25, 
                                    self.ang_vel_command, 
                                    self.base_height_command*2.0,
                                    self.lin_vel_command,
                                    self.stand_command,
                                    self.waist_dofs_command,
                                    # cos_phase,
                                    dof_pos_minus_default, 
                                    dof_vel*0.05,
                                    history_actor,
                                    # phase_time,
                                    projected_gravity,
                                    self.ref_upper_dof_pos,
                                    # sin_phase,
                                    # self.left_ee_force*0.1,
                                    # self.right_ee_force*0.1
                                    ], axis=1)
            estimator_obs = np.concatenate([self.last_policy_action, 
                                    base_ang_vel*0.25, 
                                    self.ang_vel_command, 
                                    self.base_height_command*2.0,
                                    self.lin_vel_command,
                                    self.stand_command,
                                    self.waist_dofs_command,
                                    # cos_phase,
                                    dof_pos_minus_default, 
                                    # dof_vel*0.05,
                                    history_estimator,
                                    self.left_ee_comp*0.1,
                                    # phase_time,
                                    projected_gravity,
                                    self.ref_upper_dof_pos,
                                    self.right_ee_comp*0.1,
                                    # sin_phase
                                    ], axis=1)
        else:
            actor_obs = np.concatenate([self.last_policy_action, 
                                # self.apply_force*0.1,
                                base_ang_vel*0.25, 
                                self.ang_vel_command, 
                                self.base_height_command*2.0,
                                self.lin_vel_command, 
                                self.stand_command,
                                self.waist_dofs_command,
                                # cos_phase,
                                dof_pos_minus_default, 
                                dof_vel*0.05,
                                # phase_time,
                                projected_gravity,
                                self.ref_upper_dof_pos,
                                # sin_phase,
                                self.left_ee_force*0.1,
                                self.right_ee_force*0.1
                                ], axis=1)
            estimator_obs = np.concatenate([self.last_policy_action,
                                # self.apply_force*0.1,
                                base_ang_vel*0.25, 
                                self.ang_vel_command, 
                                self.base_height_command*2.0,
                                self.lin_vel_command, 
                                self.stand_command,
                                self.waist_dofs_command,
                                # cos_phase,
                                dof_pos_minus_default, 
                                # dof_vel*0.05,
                                self.left_ee_comp*0.1,
                                # phase_time,
                                projected_gravity,
                                self.ref_upper_dof_pos,
                                self.right_ee_comp*0.1,
                                # sin_phase,
                                ], axis=1)
        if self.history_handler:
            self.history_handler.add("base_ang_vel", base_ang_vel*self.obs_scales["base_ang_vel"])
            self.history_handler.add("command_lin_vel", self.lin_vel_command*self.obs_scales["command_lin_vel"])
            self.history_handler.add("command_ang_vel", self.ang_vel_command*self.obs_scales["command_ang_vel"])
            self.history_handler.add("command_stand", self.stand_command*self.obs_scales["command_stand"])
            self.history_handler.add("command_base_height", self.base_height_command*self.obs_scales["command_base_height"])
            self.history_handler.add("command_waist_dofs", self.waist_dofs_command*self.obs_scales["command_waist_dofs"])
            self.history_handler.add("dof_pos", dof_pos_minus_default*self.obs_scales["dof_pos"])
            self.history_handler.add("dof_vel", dof_vel*self.obs_scales["dof_vel"])
            self.history_handler.add("projected_gravity", projected_gravity*self.obs_scales["projected_gravity"])
            self.history_handler.add("ref_upper_dof_pos", self.ref_upper_dof_pos*self.obs_scales["ref_upper_dof_pos"])
            self.history_handler.add("actions", self.last_policy_action*self.obs_scales["actions"])
            # self.history_handler.add("phase_time", phase_time*self.obs_scales["phase_time"])
            # self.history_handler.add("sin_phase", sin_phase*self.obs_scales["sin_phase"])
            # self.history_handler.add("cos_phase", cos_phase*self.obs_scales["cos_phase"])
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
        return {"actor_obs": actor_obs.astype(np.float32), "estimator_obs": estimator_obs.astype(np.float32)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    parser.add_argument('--avp_config', type=str, default='config/vision_pro.yaml', help='avp config file')
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--use_jit', action='store_true', default=False, help='use jit')
    parser.add_argument('--use_mocap', action='store_true', default=False, help='use mocap')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.avp_config) as file:
        avp_config = yaml.load(file, Loader=yaml.FullLoader)

    policy = DecoupledLocomotionManipulationForcePolicy(config=config, 
                                                        avp_config=avp_config,
                                                        model_path=args.model_path, 
                                                        use_jit=args.use_jit,
                                                        rl_rate=50, 
                                                        decimation=4)
    policy.run()