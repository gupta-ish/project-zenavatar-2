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
from decoupled_loco_manipulation_force_wocomp import DecoupledLocomotionManipulationForcePolicy
from multiprocessing import shared_memory, Array, Lock
from sim2real.teleop.image_server.image_client import ImageClient
from sim2real.teleop.open_television.tv_wrapper import TeleVisionWrapper

from utils.robot_arm_ik import G1_29_ArmIK
from utils.EE_motion_planner import EE_MotionPlanner
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
        self.init_upper_body_controller()
        self.init_motion_planner()

        self.recording_joint_data = False
        self.joint_data_log = []
        self.record_lock = threading.Lock()

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
        
        
    def init_tv(self, avp_config):
        img_config = avp_config['img_config']
        server_address = avp_config['server_address']
        image_show = avp_config['image_show']
        # Get the image from the robot's head camera (Same as the image server)
        ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
        if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
            BINOCULAR = True
        else:
            BINOCULAR = False
        if 'wrist_camera_type' in img_config:
            WRIST = True
        else:
            WRIST = False
        if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
            self.tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
        else:
            self.tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

        self.tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(self.tv_img_shape) * np.uint8().itemsize)
        self.tv_img_array = np.ndarray(self.tv_img_shape, dtype = np.uint8, buffer = self.tv_img_shm.buf)
        if WRIST:
            self.wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
            self.wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(self.wrist_img_shape) * np.uint8().itemsize)
            self.wrist_img_array = np.ndarray(self.wrist_img_shape, dtype = np.uint8, buffer = self.wrist_img_shm.buf)
            self.img_client = ImageClient(image_show = image_show, server_address=server_address,
                                          tv_img_shape = self.tv_img_shape, tv_img_shm_name = self.tv_img_shm.name, 
                                          wrist_img_shape = self.wrist_img_shape, wrist_img_shm_name = self.wrist_img_shm.name)
        else:
            self.img_client = ImageClient(image_show = image_show, server_address=server_address,
                                          tv_img_shape = self.tv_img_shape, tv_img_shm_name = self.tv_img_shm.name)
        
        self.image_receive_thread = threading.Thread(target = self.img_client.receive_process, daemon = True)
        self.image_receive_thread.start()

        # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        self.tv_wrapper = TeleVisionWrapper(BINOCULAR, self.tv_img_shape, self.tv_img_shm.name)

        # Initialize the TV data
        # --------------------------------head-------------------------------------
        #    rxx rxy rxz
        #    ryx ryy ryz
        #    rzx rzy rzz
        self.head_rmat = np.zeros((3, 3))
        # --------------------------------wrist-------------------------------------
        #    rxx rxy rxz tx 
        #    ryx ryy ryz ty 
        #    rzx rzy rzz tz 
        #      0   0   0  1  
        self.left_wrist = np.zeros((4, 4))
        self.right_wrist = np.zeros((4, 4))
        # --------------------------------hand-------------------------------------
        #    [x0, y0, z0]
        #    [x1, y1, z1]
        #    ···
        #    [x23,y23,z23] 
        #    [x24,y24,z24]  
        self.left_hand = np.zeros((25, 3))
        self.right_hand = np.zeros((25, 3))
    
    def run(self):
        total_inference_cnt = 0
        start_time = time.time()
        try:
            self.running_tv = False
            while True:
                if self.use_joystick and self.wc_msg is not None:
                    self.process_joystick_input()
                if self.running_tv:
                    self.head_rmat, self.left_wrist, self.right_wrist, self.left_hand, self.right_hand = self.tv_wrapper.get_data()
                    self.EE_left_R = self.left_wrist[0:3, 0:3]
                    self.EE_right_R = self.right_wrist[0:3, 0:3]
                    self.EE_left_x = self.left_wrist[0, 3] - 0.05
                    self.EE_right_x = self.right_wrist[0, 3] - 0.05
                    self.EE_left_y = self.left_wrist[1, 3] - 0.05
                    self.EE_right_y = self.right_wrist[1, 3] + 0.05
                    self.EE_left_z = self.left_wrist[2, 3]
                    self.EE_right_z = self.right_wrist[2, 3]
                    self.update_waypoints()
                    # tv_resized_image = cv2.resize(self.tv_img_array, (self.tv_img_shape[1] // 2, self.tv_img_shape[0] // 2))
                    # cv2.imshow("record image", tv_resized_image)
                    # key = cv2.waitKey(1) & 0xFF
                self.policy_action()
                end_time = time.time()
                total_inference_cnt += 1
                self.rate.sleep()
        except KeyboardInterrupt:
            pass
    
    
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
                                    # self.left_ee_comp*0.1,
                                    # phase_time,
                                    projected_gravity,
                                    self.ref_upper_dof_pos,
                                    # self.right_ee_comp*0.1,
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
                                # self.left_ee_comp*0.1,
                                # phase_time,
                                projected_gravity,
                                self.ref_upper_dof_pos,
                                # self.right_ee_comp*0.1,
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
           if self.recording_joint_data:
                with self.record_lock:
                    self.joint_data_log.append(qpos[:, -14:].squeeze().tolist())
           qvel = self.robot_state_data[:, 7+self.num_dofs+6:7+self.num_dofs+6+self.num_dofs]
           qtau = self.robot_state_data[:, 7+self.num_dofs+6+self.num_dofs:7+self.num_dofs+6+self.num_dofs+self.num_dofs]
           if self.MP_enabled == False:
               # Control upper qpos
                upper_body_qpos, upper_body_tauff = self.upper_body_controller.get_q_tau(
                    self.waypoints_left[0],
                    self.waypoints_right[0],
                    self.EE_efrc_L, self.EE_efrc_R,
                )
                self.ref_upper_dof_pos[:, -14:] = upper_body_qpos
                # Control upper tau
                cmd_tau[-14:] = upper_body_tauff
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
        scaled_policy_action, *_  = self.rl_inference(self.robot_state_data)
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

    def handle_joystick_button(self, cur_key):
        super().handle_joystick_button(cur_key)
        if cur_key == "X+Y":
            self.running_tv = not self.running_tv
            self.logger.info(f"TV running: {self.running_tv}")
        if self.upper_body_controller:
            if cur_key == "R1+up":
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
            # elif cur_key == "L1":
            #     if self.MP_enabled == True:
            #         self.MP_enabled = False
            #         self.waypoint_index = 0
            #         self.logger.info(colored("Motion Planner Disabled", "red"))
            #     else:
            #         self.MP_enabled = True
            #         # TODO: motion planner
            #         # self.MP_waypoints_left, self.MP_waypoints_right = self.MP.generate_waypoints()
            #         self.logger.info(colored("Motion Planner Enabled", "green"))
            elif cur_key == "L1":
                with self.record_lock:
                    self.recording_joint_data = not self.recording_joint_data
                    if self.recording_joint_data:
                        self.joint_data_log = []
                        self.logger.info(colored("Started recording 14-DoF joint positions", "cyan"))
                    else:
                        self.logger.info(colored("Stopped recording. Saving data...", "cyan"))
                        import csv
                        import os
                        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                        save_dir = os.path.join(repo_root, "utils", "data")
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, "recorded_joint_positions_4.csv")
                        with open(save_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerows(self.joint_data_log)
                        self.logger.info(colored(f"Saved to {save_path}", "green"))

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