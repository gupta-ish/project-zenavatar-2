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
from multiprocessing import shared_memory

from sim2real.utils.util import quat_rotate_numpy, quat_rotate_inverse_numpy
# from utils.motion_planning import MotionPlanning

##For PiVicon
from scipy.signal import butter, lfilter
from pyvicon_datastream import tools
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32
import math

import os


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
                 vicon_config,
                 vicon_config_table_pick,
                 vicon_config_table_drop,
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
        
        self.vicon_config = vicon_config
        self.vicon_g1 = Vicon(self.vicon_config)
        self.vicon_config_table_pick = vicon_config_table_pick
        self.vicon_table_pick = Vicon(self.vicon_config_table_pick)

        self.vicon_config_table_drop = vicon_config_table_drop
        self.vicon_table_drop = Vicon(self.vicon_config_table_drop)
        
        self.waist_dofs_command = np.zeros((1, 3))
        self.init_upper_body_controller()
        self.init_motion_planner()
        self.st = 0
        self.toggle_forward = False
        self.joystick_override_enabled = True
        self.sim = True
        self.reach = False
        self.test_ = False
        self.turn = True
        self.flag_of_moving = False
        self.flag_of_lateral_complete = False
        self.picked = False
        self.picked2 = False
        self.picked3 = False
        self.count = 0

        # _, self.position_init, _, _, _ = self.vicon_g1.get_vicon_data()
        # os.system("sudo bash ../../live-pose-FastSAM/docker/run_container.sh")

        # os.system('gnome-terminal -- bash -c "sudo bash ../../live-pose-FastSAM/docker/run_container.sh; exec bash"')

        # # Step 2: Wait a few seconds to ensure the container is running
        # time.sleep(5)

        # print("Container started successfully.")
        # breakpoint()

        # # Step 3: Run your desired command inside the container
        # os.system('sudo docker exec -it foundationpose bash -c "cd /home/zyh/Ishita/project-zenavatar/live-pose-FastSAM && bash run_live.sh"')



        # Path variables
        # container_script = "../../live-pose-FastSAM/docker/run_container.sh"
        # command_inside_container = "cd /home/zyh/Ishita/project-zenavatar/live-pose-FastSAM && bash run_live.sh"

        # # Launch everything in a new terminal
        # os.system(f'gnome-terminal -- bash -c "sudo bash {container_script} && sudo docker exec -it foundationpose bash -c \'{command_inside_container}\'; exec bash"')

    def init_upper_body_controller(self):
        self.upper_body_controller = G1_29_ArmIK(Unit_Test=False, Visualization=False)
        self.waypoint_index = 0
        self.speed_factor = 0.05
        self.base_z_offset = 0.8
        # Initialize waypoints
        self.degrees = -10
        self.theta = np.radians(self.degrees)
        self.EE_left_R = np.array([[np.cos(-self.theta), -np.sin(-self.theta), 0],
                                   [np.sin(-self.theta),  np.cos(-self.theta), 0],
                                   [                 0,                   0, 1]])
        self.EE_right_R = np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
                                   [np.sin(self.theta),  np.cos(self.theta), 0],
                                   [                 0,                   0, 1]])
        self.EE_left_x = 0.25#0.3
        self.EE_right_x = 0.25#0.3
        self.EE_left_y = 0.28#0.23
        self.EE_right_y = -0.28#-0.23
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
                                  pin.SE3(np.eye(3), np.array([0.35, 0.38 , 0.2])), ]
        self.MP_waypoints_right = [pin.SE3(np.eye(3), np.array([0.25, -0.18 , 0.08])),
                                   pin.SE3(np.eye(3), np.array([0.35, -0.18 , 0.08])), 
                                   pin.SE3(np.eye(3), np.array([0.35, -0.38 , 0.2])), ]
        
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

    # def init_motion_planning(self):
    #     self.motion_planning_obj = MotionPlanning()

    def state_index(self):
        self.st = self.st + 1

    def read_trajectory_data(self, filename, num_lines=100):
        """
        Read the last `num_lines` of trajectory data from the specified file.
        Returns arrays of coordinates for both points and object center.
        """
        pt1_x, pt1_y, pt1_z = [], [], []
        pt2_x, pt2_y, pt2_z = [], [], []
        obj_x, obj_y, obj_z = [], [], []  # Object center coordinates in robot frame

        with open(filename, 'r') as f:
            lines = f.readlines()[-num_lines:]  # Read only the last `num_lines` lines

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Process object center in robot frame
            if line.startswith('# object center in robot frame'):
                if i + 3 < len(lines):
                    obj_x.append(float(lines[i + 1].strip()))
                    obj_y.append(float(lines[i + 2].strip()))
                    obj_z.append(float(lines[i + 3].strip()))
                    i += 4  # Skip processed lines

            # Process pt_transformed1
            elif line.startswith('# pt_transformed1'):
                if i + 3 < len(lines):
                    pt1_x.append(float(lines[i + 1].strip()))
                    pt1_y.append(float(lines[i + 2].strip()))
                    pt1_z.append(float(lines[i + 3].strip()))
                    i += 4  

            # Process pt_transformed2
            elif line.startswith('# pt_transformed2'):
                if i + 3 < len(lines):
                    pt2_x.append(float(lines[i + 1].strip()))
                    pt2_y.append(float(lines[i + 2].strip()))
                    pt2_z.append(float(lines[i + 3].strip()))
                    i += 4  
            else:
                i += 1  # Move to the next line if no match

        return (np.array(pt1_x), np.array(pt1_y), np.array(pt1_z)), \
            (np.array(pt2_x), np.array(pt2_y), np.array(pt2_z)), \
            (np.array(obj_x), np.array(obj_y), np.array(obj_z))

    def reach_pos(self):
        
        #############################--ROBOT POSE FROM REAL--####################################################
        # currently getting pose from vicon
        current_time, position, roll_g1, pitch_g1, yaw_g1 = self.vicon_g1.get_vicon_data()

        if current_time is None:
            print("Warning: No Vicon data available")
            return

        
        pos_g1 = position 
        x, y = pos_g1[0], pos_g1[1]          
        orien_g1 = np.array([roll_g1, pitch_g1, yaw_g1])
        yaw = orien_g1[2]
        # yaw = np.radians(yaw)

        print(f"robot position: {pos_g1}")
        print(f"robot orientation: {orien_g1}")

        current_time_table, table_position, r_table, p_table, y_table = self.vicon_table_pick.get_vicon_data()

        # if current_time_table is None:
        #     print("Warning: No Vicon data from table available")
        #     return
        print( "Table Position:", table_position)
        print( "Table Orientation:", y_table)
        x_goal = table_position[0]
        y_goal = table_position[1] - 0.4
        # # theta_goal_deg = y_table  

        ##############################--ROBOT POSE FROM SIM--#######################################################

        # shm = shared_memory.SharedMemory(name='pose_of_G1', create=False) # this gets robot pose from simulation continuously
        # self.pose_mujoco = np.ndarray((6,), dtype=np.float64, buffer=shm.buf)

        # print("Robot Position:", self.pose_mujoco[:3])
        # # print("Robot RPY:", self.pose_mujoco[3:])

        # # print(f"robot position: {pos_g1}")
        # # print(f"robot orientation: {orien_g1}")

        # pos_g1 = self.pose_mujoco[:3]
        # orien_g1 = self.pose_mujoco[3:]
        # x, y = pos_g1[0], pos_g1[1]
        # yaw_deg = orien_g1[2]
        # yaw = np.radians(yaw_deg)

        # print("Robot Y:", yaw)

        # # Goal position
        # x_goal = 2
        # y_goal = 0
        
        # currently getting pose from simulation
        # pos_g1 = self.pose_mujoco[:3]
        # orien_g1 = self.pose_mujoco[3:]
        # x_goal = 2
        # y_goal = 2
        # theta_goal_deg = 0
        #####################################################################################
        
        # # currently hardcode the goal position (PICKUP-TABLE POSE)
        
        
            
        # theta_goal = np.radians(theta_goal_deg)
        # # print(f"x_goal: {x_goal}, y_goal: {y_goal}, theta_goal_deg: {theta_goal_deg}")
        # print(f"goal position: {x_goal}, {y_goal}")

        # angle_threshold = 0.1 #rad  # 0.1 --> ~6 degrees
        # distance_threshold = 0.2 #m
        # lateral_threshold = 0.1

        # yaw = orien_g1[2]
        # yaw = np.radians(yaw)  # Convert to radians

        # orientation_error = theta_goal - yaw
        # orientation_error = np.arctan2(np.sin(orientation_error), np.cos(orientation_error))  # Normalize to [-pi, pi]

        # if abs(orientation_error) > angle_threshold and self.turn:
        #     if orientation_error > 0:
        #         self.ang_vel_command[0, 0] = 0.3  # turn left
        #         print("Rotating left to align with goal yaw")
        #     else:
        #         self.ang_vel_command[0, 0] = -0.3  # turn right
        #         print("Rotating right to align with goal yaw")
        # elif abs(orientation_error) <= angle_threshold and self.turn:
        #     self.lin_vel_command[0, :] = 0.
        #     self.ang_vel_command[0, :] = 0.
        #     self.turn = False
        #     print("Goal reached orientation — stopping")
        

        # dx = x_goal - pos_g1[0]
        # dy = y_goal - pos_g1[1]
        # cos_yaw = np.cos(-yaw)
        # sin_yaw = np.sin(-yaw)
        
        # dx_robot = cos_yaw * dx - sin_yaw * dy
        # dy_robot = sin_yaw * dx + cos_yaw * dy

        # # dx_robot = -dy_robot
        # # dy_robot = dx_robot

        # # # Angle to target in robot frame
        # # # theta = np.arctan2(dy_robot, dx_robot)

        # angle_error = np.arctan2(dy_robot, dx_robot)
        # pos_error = np.hypot(dx_robot, dy_robot)
        # lateral_error = dy_robot


        # # if abs(orientation_error) > angle_threshold and self.turn != True:
        # #     if orientation_error > 0:
        # #         self.ang_vel_command[0, 0] = 0.1  # turn left
        # #         print("Rotating left to align with goal yaw -2")
        # #     else:
        # #         self.ang_vel_command[0, 0] = -0.1  # turn right
        # #         print("Rotating right to align with goal yaw -2")
        # # elif abs(orientation_error) <= angle_threshold and self.turn != True:
        # #     self.lin_vel_command[0, :] = 0.
        # #     self.ang_vel_command[0, :] = 0.
        # #     print("Goal reached orientation — stopping - 2")


        # if abs(angle_error) > angle_threshold and self.turn != True:
        #     if angle_error > 0:
        #         self.ang_vel_command[0, 0] = 0.1  # turn left
        #         print("Turning left") 
        #     else:
        #         self.ang_vel_command[0, 0] = -0.1  # turn right
        #         print("Turning right")

        # if abs(angle_error) <= angle_threshold and self.turn != True:
        #     self.turn = False
        #     self.lin_vel_command[0, :] = 0.
        #     self.ang_vel_command[0, :] = 0.

        # # if abs(lateral_error) > lateral_threshold and self.turn != True:
        # #     if lateral_error > 0:
        # #         self.lin_vel_command[0, 1] = 0.1  # move right (in robot frame)
        # #         print("Moving right")
        # #     else:
        # #         self.lin_vel_command[0, 1] = -0.1  # move left
        # #         print("Moving left")

        # # elif abs(lateral_error) <= lateral_threshold and self.turn != True:
        # #     self.lin_vel_command[0, :] = 0.
        # #     self.ang_vel_command[0, :] = 0.
        # #     self.flag_of_lateral_complete = True
        # #     print("Goal reached lateral — stopping")


        # ############

        # if pos_error > distance_threshold and self.turn != True:
        #     self.lin_vel_command[0, 0] = 0.1  # move forward
        #     print("Moving forward")
        # elif pos_error <= distance_threshold and self.turn != True:
        #     self.lin_vel_command[0, :] = 0.
        #     self.ang_vel_command[0, :] = 0.
        #     self.flag_of_moving = True
        #     print("Goal reached pose — stopping")

        # if self.flag_of_moving and self.flag_of_lateral_complete:
        #     self.reach = False
        #     self.stand_command = 1 - self.stand_command
        #     if self.stand_command == 0:
        #         self.ang_vel_command[0, 0] = 0.
        #         self.lin_vel_command[0, 0] = 0.
        #         self.lin_vel_command[0, 1] = 0.
        #         self.logger.info(colored("Stance command", "blue"))
        #     else:
        #         self.base_height_command[0, 0] = self.desired_base_height
        #         self.logger.info(colored("Walk command", "blue"))


    ##################################################################

        

        # Thresholds
        angle_threshold = 0.2  # radians (~6 degrees)
        distance_threshold = 0.2  # meters
        lateral_threshold = 0.05

        # --- Step 1: Face +X direction (yaw = 0) ---
        goal_yaw = 1.75  # Always want to face +X

        # Compute shortest rotation needed to make yaw → 0
        yaw_error = np.arctan2(np.sin(goal_yaw - yaw), np.cos(goal_yaw - yaw))
        # yaw_error = goal_yaw - yaw
        print(" YAW error: ", yaw_error)

        # position_error_X = position[0] - self.position_init[0]
        # print("Position error X: ", position_error_X)
        # position_error_Y = position[1] - self.position_init[1]
        # print("Position error Y: ", position_error_Y)

        if abs(yaw_error) > angle_threshold and not self.flag_of_moving:
            # Rotate in place to face +X
            self.lin_vel_command[0, :] = -0.2
            if yaw_error > 0:
                self.ang_vel_command[0, 0] = 1
                print("Rotating left to face +X")
            else:
                self.ang_vel_command[0, 0] = -0.5
                print("Rotating right to face +X")

            # self.lin_vel_command[0, 0] = position_error_X * 0.1
            # self.lin_vel_command[0, 1] = position_error_Y * 0.1

        else:
            # Yaw is aligned with +X, start moving toward goal
            self.ang_vel_command[0, 0] = 0.17

            dx = x_goal - x
            dy = y_goal - y
            distance = math.hypot(dx, dy)

            ####################

            if x > x_goal + lateral_threshold:
                self.lin_vel_command[0, 1] = 0.3  # Move left
                print("Moving left to correct lateral error")
            elif x < x_goal - lateral_threshold:
                self.lin_vel_command[0, 1] = -0.5  # Move right
                print("Moving right to correct lateral error")
            else:
                self.lin_vel_command[0, 1] = 0.

            ############

            if distance > distance_threshold:
                self.lin_vel_command[0, 0] = 0.4  # Move forward
                print("Moving forward toward goal")
            else:
                self.lin_vel_command[0, :] = 0.
                self.ang_vel_command[0, :] = 0.
                self.flag_of_moving = True
                print("Goal reached — stopping")
        
            

        # Trigger something once goal is reached
        if self.flag_of_moving:
            self.reach = False
            self.flag_of_moving = False
            # os.system("../../../live-pose-FastSAM/docker/run_container.sh")

            #### run fpose here
            self.stand_command = 1 - self.stand_command
            if self.stand_command == 0:
                self.ang_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 1] = 0.
                self.logger.info(colored("Stance command", "blue"))
            else:
                self.base_height_command[0, 0] = self.desired_base_height
                self.logger.info(colored("Walk command", "blue"))

            self.logger.info("Reached goal point. You can trigger next state here.")

    def reach_pos_2(self):
        
       
        
            #############################--ROBOT POSE FROM REAL--####################################################
        # currently getting pose from vicon
        current_time, position, roll_g1, pitch_g1, yaw_g1 = self.vicon_g1.get_vicon_data()

        if current_time is None:
            print("Warning: No Vicon data available")
            return

        
        pos_g1 = position 
        x, y = pos_g1[0], pos_g1[1]          
        orien_g1 = np.array([roll_g1, pitch_g1, yaw_g1])
        yaw = orien_g1[2]
        # yaw = np.radians(yaw)

        print(f"robot position: {pos_g1}")
        print(f"robot orientation: {orien_g1}")

        current_time_table, table_position, r_table, p_table, y_table = self.vicon_table_pick.get_vicon_data()

        # if current_time_table is None:
        #     print("Warning: No Vicon data from table available")
        #     return
        print( "Table Position:", table_position)
        print( "Table Orientation:", y_table)
        x_goal = table_position[0]
        y_goal = table_position[1] - 1.5
        # # theta_goal_deg = y_table  
        print("y_goal:- ", y_goal)

        

        # Thresholds
        angle_threshold = 0.3  # radians (~6 degrees)
        distance_threshold = 0.2  # meters
        lateral_threshold = 0.05

        # --- Step 1: Face +X direction (yaw = 0) ---
        goal_yaw = 1.57  # Always want to face +X

        # Compute shortest rotation needed to make yaw → 0
        yaw_error = np.arctan2(np.sin(goal_yaw - yaw), np.cos(goal_yaw - yaw))
        # yaw_error = goal_yaw - yaw
        # print(" YAW error: ", yaw_error)

        if abs(yaw_error) > angle_threshold and not self.flag_of_moving:
            # Rotate in place to face +X
            self.lin_vel_command[0, :] = -0.2
            if yaw_error > 0:
                self.ang_vel_command[0, 0] = 1
                print("Rotating left to face +X")
            else:
                self.ang_vel_command[0, 0] = -0.5
                print("Rotating right to face +X")

            # self.lin_vel_command[0, 0] = position_error_X * 0.1
            # self.lin_vel_command[0, 1] = position_error_Y * 0.1

        else:
            # Yaw is aligned with +X, start moving toward goal
            self.ang_vel_command[0, 0] = 0.17

            dx = x_goal - x
            dy = y_goal - y
            distance = math.hypot(dx, dy)

            ####################

            if x > x_goal + lateral_threshold:
                self.lin_vel_command[0, 1] = 0.3  # Move left
                print("Moving left to correct lateral error")
            elif x < x_goal - lateral_threshold:
                self.lin_vel_command[0, 1] = -0.5  # Move right
                print("Moving right to correct lateral error")
            else:
                self.lin_vel_command[0, 1] = 0.

            ############

            if distance > distance_threshold:
                self.lin_vel_command[0, 0] = -0.3  # Move backward
                print("Moving backward toward goal")
            else:
                self.lin_vel_command[0, :] = 0.
                self.ang_vel_command[0, :] = 0.
                self.flag_of_moving = True
                print("Goal reached — stopping")
            
                

            # Trigger something once goal is reached
        if self.flag_of_moving:
            # self.flag_of_moving = False
            self.picked = False
            self.flag_of_moving = False
            # self.picked2 = True
            self.stand_command = 1 - self.stand_command
            if self.stand_command == 0:
                self.ang_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 1] = 0.
                self.logger.info(colored("Stance command", "blue"))
            else:
                self.base_height_command[0, 0] = self.desired_base_height
                self.logger.info(colored("Walk command", "blue"))

            self.logger.info("Reached goal point. You can trigger next state here.")


    def reach_pos_3(self):
        # if self.picked2 == True:
        
        #############################--ROBOT POSE FROM REAL--####################################################
        # currently getting pose from vicon
        current_time, position, roll_g1, pitch_g1, yaw_g1 = self.vicon_g1.get_vicon_data()

        if current_time is None:
            print("Warning: No Vicon data available")
            return

        
        pos_g1 = position 
        x, y = pos_g1[0], pos_g1[1]          
        orien_g1 = np.array([roll_g1, pitch_g1, yaw_g1])
        yaw = orien_g1[2]
        # yaw = np.radians(yaw)

        print(f"robot position: {pos_g1}")
        print(f"robot orientation: {orien_g1}")

        current_time_table, table_position, r_table, p_table, y_table = self.vicon_table_drop.get_vicon_data()

        # if current_time_table is None:
        #     print("Warning: No Vicon data from table available")
        #     return
        print( "Table Position:", table_position)
        print( "Table Orientation:", y_table)
        x_goal = table_position[0] - 0.5
        y_goal = table_position[1] - 0.4
        # # theta_goal_deg = y_table  

        

        # Thresholds
        angle_threshold = 0.2  # radians (~6 degrees)
        distance_threshold = 0.2  # meters
        lateral_threshold = 0.1

        # --- Step 1: Face +X direction (yaw = 0) ---
        goal_yaw = -0.19  # Always want to face +X

        # Compute shortest rotation needed to make yaw → 0
        yaw_error = np.arctan2(np.sin(goal_yaw - yaw), np.cos(goal_yaw - yaw))
        # yaw_error = goal_yaw - yaw
        print(" YAW error: ", yaw_error)

        if abs(yaw_error) > angle_threshold and not self.flag_of_moving:
            # Rotate in place to face +X
            self.lin_vel_command[0, :] = -0.2
            if yaw_error > 0:
                self.ang_vel_command[0, 0] = 1
                print("reach_pos_3: Rotating left to face +X")
            else:
                self.ang_vel_command[0, 0] = -0.5
                print("reach_pos_3: Rotating right to face +X")

            # self.lin_vel_command[0, 0] = position_error_X * 0.1
            # self.lin_vel_command[0, 1] = position_error_Y * 0.1

        else:
            # Yaw is aligned with +X, start moving toward goal
            self.ang_vel_command[0, 0] = 0.17

            dx = x_goal - x
            dy = y_goal - y
            distance = math.hypot(dx, dy)

            ####################

            if abs(y) > abs(y_goal) + lateral_threshold:
                self.lin_vel_command[0, 1] = 0.3  # Move left
                print("reach_pos_3: Moving left to correct lateral error")
            elif abs(y) < abs(y_goal) - lateral_threshold:
                self.lin_vel_command[0, 1] = -0.5  # Move right
                print("reach_pos_3: Moving right to correct lateral error")
            else:
                self.lin_vel_command[0, 1] = 0.

            ############

            if distance > distance_threshold:
                self.lin_vel_command[0, 0] = 0.4  # Move forward
                print("reach_pos_3: Moving forward toward goal")
            else:
                self.lin_vel_command[0, :] = 0.
                self.ang_vel_command[0, :] = 0.
                self.flag_of_moving = True
                print("reach_pos_3: Goal reached — stopping")
        
            

        # Trigger something once goal is reached
        if self.flag_of_moving:
            self.reach = False
            self.picked = False
            self.picked2 = False
            self.flag_of_moving = False
            
            # os.system("../../../live-pose-FastSAM/docker/run_container.sh")

            #### run fpose here
            self.stand_command = 1 - self.stand_command
            if self.stand_command == 0:
                self.ang_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 1] = 0.
                self.logger.info(colored("Stance command", "blue"))
            else:
                self.base_height_command[0, 0] = self.desired_base_height
                self.logger.info(colored("Walk command", "blue"))

            self.logger.info("Reached goal point. You can trigger next state here.")

    def reach_pos_4(self):
        # if self.picked2 == True:
        
        #############################--ROBOT POSE FROM REAL--####################################################
        # currently getting pose from vicon
        current_time, position, roll_g1, pitch_g1, yaw_g1 = self.vicon_g1.get_vicon_data()

        if current_time is None:
            print("Warning: No Vicon data available")
            return

        
        pos_g1 = position 
        x, y = pos_g1[0], pos_g1[1]          
        orien_g1 = np.array([roll_g1, pitch_g1, yaw_g1])
        yaw = orien_g1[2]
        # yaw = np.radians(yaw)

        print(f"robot position: {pos_g1}")
        print(f"robot orientation: {orien_g1}")

        current_time_table, table_position, r_table, p_table, y_table = self.vicon_table_drop.get_vicon_data()

        # if current_time_table is None:
        #     print("Warning: No Vicon data from table available")
        #     return
        print( "Table Position:", table_position)
        print( "Table Orientation:", y_table)
        x_goal = table_position[0] - 1.4
        y_goal = table_position[1] - 0.4
        # # theta_goal_deg = y_table  

        

        # Thresholds
        angle_threshold = 0.2  # radians (~6 degrees)
        distance_threshold = 0.2  # meters
        lateral_threshold = 0.2

        # --- Step 1: Face +X direction (yaw = 0) ---
        goal_yaw = 0  # Always want to face +X

        # Compute shortest rotation needed to make yaw → 0
        yaw_error = np.arctan2(np.sin(goal_yaw - yaw), np.cos(goal_yaw - yaw))
        # yaw_error = goal_yaw - yaw
        print(" YAW error: ", yaw_error)

        if abs(yaw_error) > angle_threshold and not self.flag_of_moving:
            # Rotate in place to face +X
            self.lin_vel_command[0, :] = -0.2
            if yaw_error > 0:
                self.ang_vel_command[0, 0] = 1
                print("reach_pos_4: Rotating left to face +X")
            else:
                self.ang_vel_command[0, 0] = -0.5
                print("reach_pos_4: Rotating right to face +X")

            # self.lin_vel_command[0, 0] = position_error_X * 0.1
            # self.lin_vel_command[0, 1] = position_error_Y * 0.1

        else:
            # Yaw is aligned with +X, start moving toward goal
            self.ang_vel_command[0, 0] = 0.17

            dx = x_goal - x
            dy = y_goal - y
            distance = math.hypot(dx, dy)

            ####################

            if abs(y) > abs(y_goal) + lateral_threshold:
                self.lin_vel_command[0, 1] = 0.3  # Move left
                print("reach_pos_4: Moving left to correct lateral error")
            elif abs(y) < abs(y_goal) - lateral_threshold:
                self.lin_vel_command[0, 1] = -0.5  # Move right
                print("reach_pos_4: Moving right to correct lateral error")
            else:
                self.lin_vel_command[0, 1] = 0.

            ############

            if distance > distance_threshold:
                self.lin_vel_command[0, 0] = -0.2  # Move forward
                print("reach_pos_4: Moving backward toward goal")
            else:
                self.lin_vel_command[0, :] = 0.
                self.ang_vel_command[0, :] = 0.
                self.flag_of_moving = True
                print("reach_pos_4: Goal reached — stopping")
        
            

        # Trigger something once goal is reached
        if self.flag_of_moving:
            self.reach = False
            self.picked = False
            self.picked2 = False
            self.picked3 = False
            
            # os.system("../../../live-pose-FastSAM/docker/run_container.sh")

            #### run fpose here
            self.stand_command = 1 - self.stand_command
            if self.stand_command == 0:
                self.ang_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 1] = 0.
                self.logger.info(colored("Stance command", "blue"))
            else:
                self.base_height_command[0, 0] = self.desired_base_height
                self.logger.info(colored("Walk command", "blue"))

            self.logger.info("Reached goal point. You can trigger next state here.")


        


    def run_motion_planning(self):
        self.robot_state_data = self.state_processor._prepare_low_state()

        qpos = self.robot_state_data[:, 7:7+self.num_dofs]
        # self.upper_body_controller.debug()
        L_ee_pose, R_ee_pose = self.upper_body_controller.get_end_effector_poses(qpos[:, -14:])

        # print("L_ee_pose-->> ", L_ee_pose)
        # print("R_ee_pose---->> ", R_ee_pose)
        # comes from perception
        # goal_pose = [0.5, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]

        
        # goal_pose_L = [0.298911, 0.280625, 0.092339]
        # goal_pose_R = [0.293747, -0.299146, 0.087083]
        print("self.st------->>>>>", self.st)

        fpose_file_path = "/home/zyh/Praj/live-pose-FastSAM/FoundationPose/transformed_points.txt"

        self.degrees = 25
        self.theta = np.radians(self.degrees)
        self.EE_left_R = np.array([[np.cos(-self.theta), -np.sin(-self.theta), 0],
                                [np.sin(-self.theta),  np.cos(-self.theta), 0],
                                [                 0,                   0, 1]])
        self.EE_right_R = np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
                                [np.sin(self.theta),  np.cos(self.theta), 0],
                                [                 0,                   0, 1]])

        old_pt1 = np.array([0.32, 0.25, 0.05])
        old_pt2 = np.array([0.32, -0.25, 0.05]) 

        old_pt1_tup = (np.array([old_pt1[0]]), np.array([old_pt1[1]]), np.array([old_pt1[2]]))
        old_pt2_tup = (np.array([old_pt2[0]]), np.array([old_pt2[1]]), np.array([old_pt2[2]]))

         
        if self.st == 1:
            print(f'reading traj data from {fpose_file_path}')
            check_box = True
            pt1_coords, pt2_coords, obj_coords = self.read_trajectory_data(fpose_file_path, num_lines=23)
            if (pt1_coords[0] > 1.5 or pt1_coords[1] > 1.5 or pt1_coords[2] > 0.8 or pt2_coords[0] > 1.5 or pt2_coords[1] > 0.2 or pt2_coords[2] > 0.8 \
                or pt1_coords[0] < 0 or pt1_coords[1] < -0.2 or pt1_coords[2] < -0.5 or pt2_coords[0] < 0 or pt2_coords[1] < -1.5 or pt2_coords[2] < -0.5):
                check_box = False
                print("check_box: ", check_box)

            if (len(pt2_coords[0]) > 0 and len(pt1_coords[0]) and (check_box)) > 0:
                if pt2_coords[1]<0:
                    pass
                elif pt1_coords[1]<0:
                    pt1_temp = pt1_coords
                    pt1_coords = pt2_coords
                    pt2_coords = pt1_temp
            else:
                pt1_coords, pt2_coords = old_pt1_tup, old_pt2_tup


            print("pt1 ",pt1_coords, "    pt2", pt2_coords)
            print("pt1 and pt2 size: ", len(pt1_coords[0]), len(pt2_coords[0]))
            #reach
            # self.goal_pose_L_1 = pin.SE3(np.eye(3), np.array([pt1_coords[0]+0.05, pt1_coords[1], pt1_coords[2]]))
            # self.goal_pose_R_1 = pin.SE3(np.eye(3), np.array([pt2_coords[0]+0.05, pt2_coords[1], pt2_coords[2]]))
            self.goal_pose_L_1 = pin.SE3(self.EE_left_R, np.array([pt1_coords[0]+0.05, pt1_coords[1], pt1_coords[2]-0.02]))
            self.goal_pose_R_1 = pin.SE3(self.EE_right_R, np.array([pt2_coords[0]+0.05, pt2_coords[1], pt2_coords[2]-0.02]))
            left_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(L_ee_pose, self.goal_pose_L_1, 2)
            right_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(R_ee_pose, self.goal_pose_R_1, 2)
            old_pt1, old_pt2 = pt1_coords, pt2_coords
            self.MP_waypoints_left = left_ee_trajectory
            self.MP_waypoints_right = right_ee_trajectory
        # qpos = self.robot_state_data[:, 7:7+self.num_dofs]

        elif self.st == 2:
            #grasp
            # 36
            temp_goal_pose_L_2_y = self.goal_pose_L_1.translation[1] - 0.1
            temp_goal_pose_R_2_y = self.goal_pose_R_1.translation[1] + 0.1

            # print("temp_goal_pose_L_2_y ",temp_goal_pose_L_2_y)
            # print("temp_goal_pose_R_2_y ",temp_goal_pose_R_2_y)

            
            # self.goal_pose_L_2 = pin.SE3(np.eye(3), np.array([self.goal_pose_L_1.translation[0], 0.03, self.goal_pose_L_1.translation[2]]))
            # self.goal_pose_R_2 = pin.SE3(np.eye(3), np.array([self.goal_pose_R_1.translation[0], -0.03, self.goal_pose_L_1.translation[2]]))

            self.goal_pose_L_2 = pin.SE3(self.EE_left_R, np.array([self.goal_pose_L_1.translation[0], 0.03, self.goal_pose_L_1.translation[2]]))
            self.goal_pose_R_2 = pin.SE3(self.EE_right_R, np.array([self.goal_pose_R_1.translation[0], -0.03, self.goal_pose_L_1.translation[2]]))
            left_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(L_ee_pose, self.goal_pose_L_2, 2)
            right_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(R_ee_pose, self.goal_pose_R_2, 2)
            self.MP_waypoints_left = left_ee_trajectory
            self.MP_waypoints_right = right_ee_trajectory

        elif self.st == 3:
            # pickup
            temp_goal_pose_L_3_y = self.goal_pose_L_2.translation[2] + 0.25
            temp_goal_pose_R_3_y = self.goal_pose_R_2.translation[2] + 0.25

            # goal_pose_L_3 = pin.SE3(np.eye(3), np.array([self.goal_pose_L_2.translation[0], self.goal_pose_L_2.translation[1], temp_goal_pose_L_3_y]))
            # goal_pose_R_3 = pin.SE3(np.eye(3), np.array([self.goal_pose_R_2.translation[0], self.goal_pose_R_2.translation[1], temp_goal_pose_R_3_y]))


            goal_pose_L_3 = pin.SE3(self.EE_left_R, np.array([self.goal_pose_L_2.translation[0], self.goal_pose_L_2.translation[1], temp_goal_pose_L_3_y]))
            goal_pose_R_3 = pin.SE3(self.EE_right_R, np.array([self.goal_pose_R_2.translation[0], self.goal_pose_R_2.translation[1], temp_goal_pose_R_3_y]))
            left_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(L_ee_pose, goal_pose_L_3, 2)
            right_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(R_ee_pose, goal_pose_R_3, 2)
            self.MP_waypoints_left = left_ee_trajectory
            self.MP_waypoints_right = right_ee_trajectory

        elif self.st == 4:
            #drop (go back to the grasp goal pose)
            # goal_pose_L_4 = pin.SE3(self.EE_left_R, np.array([0.55, self.goal_pose_L_2.translation[1], self.goal_pose_L_2.translation[2] + 0.25]))
            # goal_pose_R_4 = pin.SE3(self.EE_right_R, np.array([0.1, self.goal_pose_R_2.translation[1], self.goal_pose_R_2.translation[2] + 0.25]))
            # left_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(L_ee_pose, goal_pose_L_4, 2)
            # right_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(R_ee_pose, goal_pose_R_4, 2)
            # self.MP_waypoints_left = left_ee_trajectory
            # self.MP_waypoints_right = right_ee_trajectory

            
            self.goal_pose_L_2 = pin.SE3(self.EE_left_R, np.array([0.4, 0.03, self.goal_pose_L_1.translation[2] - 0.15]))
            self.goal_pose_R_2 = pin.SE3(self.EE_right_R, np.array([0.4, -0.03, self.goal_pose_L_1.translation[2] - 0.15]))
            left_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(L_ee_pose, self.goal_pose_L_2, 2)
            right_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(R_ee_pose, self.goal_pose_R_2, 2)
            self.MP_waypoints_left = left_ee_trajectory
            self.MP_waypoints_right = right_ee_trajectory

        elif self.st == 5:
            #unsqueeze
            self.goal_pose_L_2 = pin.SE3(self.EE_left_R, np.array([0.4, 0.2, self.goal_pose_L_1.translation[2]- 0.15]))
            self.goal_pose_R_2 = pin.SE3(self.EE_right_R, np.array([0.4, -0.2, self.goal_pose_L_1.translation[2]- 0.15]))
            left_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(L_ee_pose, self.goal_pose_L_2, 2)
            right_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(R_ee_pose, self.goal_pose_R_2, 2)
            self.MP_waypoints_left = left_ee_trajectory
            self.MP_waypoints_right = right_ee_trajectory

        else:
            return

        #pick
        # goal_pose_L = pin.SE3(np.eye(3), np.array([0.282342, 0.267832, 0.180255]))
        # goal_pose_R = pin.SE3(np.eye(3), np.array([0.278191, -0.211542, 0.200358]))
        # left_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(L_ee_pose, goal_pose_L, 2)
        # right_ee_trajectory = self.upper_body_controller.generate_waypoints_SE(R_ee_pose, goal_pose_R, 2)

        # self.MP_waypoints_left = left_ee_trajectory
        # self.MP_waypoints_right = right_ee_trajectory

        # for i, pose in enumerate(left_ee_trajectory):
        #     print(f"Left Waypoint {i} translation:", pose.translation)

        # for i, pose in enumerate(right_ee_trajectory):
        #     print(f"Right Waypoint {i} translation:", pose.translation)
    
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
                    # self.update_waypoints()
                    # tv_resized_image = cv2.resize(self.tv_img_array, (self.tv_img_shape[1] // 2, self.tv_img_shape[0] // 2))
                    # cv2.imshow("record image", tv_resized_image)
                    # key = cv2.waitKey(1) & 0xFF
                self.policy_action()
                end_time = time.time()
                total_inference_cnt += 1
                if self.reach:
                    self.reach_pos()
                if self.picked:
                    self.reach_pos_2()
                if self.picked2:
                    self.reach_pos_3()
                if self.picked3:
                    self.reach_pos_4()
                self.rate.sleep()
                # if self.test_:
                #     self.test()
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

    def test(self):
        self.lin_vel_command[0, 0] = 0.3
    
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

    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        if keycode == "f":
            self.run_motion_planning()
        # For to move the robot to the goal location
        elif keycode == "*":
            self.reach = True
            print("Reach command")
        elif keycode == "m":
            self.test_ = True
            # print("Reach commanddddddddddddddddddddddd")
        
    def handle_joystick_button(self, cur_key):
        super().handle_joystick_button(cur_key)
        if cur_key == "X+Y":
            self.running_tv = not self.running_tv
            self.logger.info(f"TV running: {self.running_tv}")
        if self.upper_body_controller:
            if cur_key == "R1+up":
                self.upper_body_controller.debug()
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
            #     self.run_motion_planning()
            elif cur_key == "L1+select":
                self.state_index()
                self.run_motion_planning()
                if self.MP_enabled == True and self.st > 5:
                    self.MP_enabled = False
                    self.st = 0
                    self.waypoint_index = 0
                    self.logger.info(colored("Motion Planner Disabled", "red"))
                    print(f"self.st is now:", self.st)
                else:
                    self.MP_enabled = True
                    # TODO: motion planner
                    # self.MP_waypoints_left, self.MP_waypoints_right = self.MP.generate_waypoints()
                    self.logger.info(colored("Motion Planner Enabled", "green"))
            elif cur_key == "B+X":
                # self.reach = True
                self.toggle_forward = not self.toggle_forward
                self.joystick_override_enabled = not self.joystick_override_enabled
                if self.toggle_forward:
                    self.reach = True
                    self.flag_of_moving = False
                    self.logger.info(colored("Starting to REACH", "green"))
                else:
                    self.flag_of_moving = True
                    self.logger.info(colored("End REACH", "red"))
            elif cur_key == "A+B":
                self.command_sender.kp_level = 1.0
                self.kp_level_lower_body = 0.0
                for i in range(15):
                    self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.kp_level_lower_body
                for i in range(15,29):
                    self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
                self.logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
                self.logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
            elif cur_key == "Y+select":
                self.toggle_forward = not self.toggle_forward
                self.joystick_override_enabled = not self.joystick_override_enabled
                if self.toggle_forward:
                    self.picked = True
                    self.flag_of_moving = False
                    self.logger.info(colored("Starting to REACH", "green"))
                else:
                    self.flag_of_moving = True
                    self.logger.info(colored("End REACH", "red"))
            elif cur_key == "B+select":
                self.toggle_forward = not self.toggle_forward
                self.joystick_override_enabled = not self.joystick_override_enabled
                if self.toggle_forward:
                    self.picked2 = True
                    self.flag_of_moving = False
                    self.logger.info(colored("Starting to REACH", "green"))
                else:
                    self.flag_of_moving = True
                    self.logger.info(colored("End REACH", "red"))
            elif cur_key == "A+select":
                self.toggle_forward = not self.toggle_forward
                self.joystick_override_enabled = not self.joystick_override_enabled
                if self.toggle_forward:
                    self.picked3 = True
                    self.flag_of_moving = False
                    self.logger.info(colored("Starting to REACH", "green"))
                else:
                    self.flag_of_moving = True
                    self.logger.info(colored("End REACH", "red"))

class Vicon():
    def __init__(self, cfg):
        
        # Vicon DataStream IP and object name
        self.vicon_tracker_ip = cfg['vicon_tracker_ip']
        self.vicon_object_name = cfg['vicon_object_name']
        self.odom_frame_id = cfg['odom_frame_id']
        self.odom_child_frame_id = cfg['odom_child_frame_id']
        self.fs = cfg['frequency']
        
        # Connect to Vicon DataStream
        self.tracker = tools.ObjectTracker(self.vicon_tracker_ip)
        if self.tracker.is_connected:
            print(f"Connected to Vicon DataStream at {self.vicon_tracker_ip}")
        else:
            print(f"Failed to connect to Vicon DataStream at {self.vicon_tracker_ip}")
            raise Exception(f"Connection to {self.vicon_tracker_ip} failed")

        # Initialize previous values for velocity computation
        self.prev_time = None
        self.prev_position = None
        self.prev_quaternion = None

        # Low-pass filter parameters
        self.cut_off_freq =  cfg['cut_off_freq']
        self.filter_order =  cfg['filter_order']
        self.filter_window_size = cfg['filter_window_size']
        self.b, self.a = butter(self.filter_order, self.cut_off_freq / (0.5 * self.fs), btype='low')

        # Initialize data buffers for filtering
        self.vel_buffer = []
        self.omega_buffer = []
                
        # Odometry publisher
        # self.odom_pub = self.create_publisher(Odometry, 'odometry', 10)
        # self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'pose', 10)
        
        # self.logger = self.get_logger()

        # Frequency counter
        self.freq_counter = 0
        # self.create_timer(1.0, self.log_frequency)
        
    def get_vicon_data(self):
        position = self.tracker.get_position(self.vicon_object_name)
        
        if not position:
            print(f"Cannot get the pose of `{self.vicon_object_name}`.")
            return None, None, None

        try:
            obj = position[2][0]
            _, _, x, y, z, roll, pitch, yaw = obj
            current_time = time.time()

            # Position and orientation
            position = np.array([x, y, z])/1000. # Convert to meters
            rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
            quaternion = rotation.as_quat()  # [x, y, z, w]

            return current_time, position, roll, pitch, yaw
        except Exception as e:
            print(f"Error retrieving Vicon data: {e}")
            return None, None, None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    parser.add_argument('--avp_config', type=str, default='config/vision_pro.yaml', help='avp config file')
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--use_jit', action='store_true', default=False, help='use jit')
    parser.add_argument('--use_mocap', action='store_true', default=False, help='use mocap')
    # args = parser.parse_args()

    # parser = argparse.ArgumentParser(description='Mocap Publisher')
    parser.add_argument('--config_vicon', type=str, default='config/mocap_g1_zenavatar.yaml', help='motion capture configuration file')
    parser.add_argument('--config_vicon_table_pick', type=str, default='config/pick_table.yaml', help='motion capture configuration file')
    parser.add_argument('--config_vicon_table_drop', type=str, default='config/drop_table.yaml', help='motion capture configuration file')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.avp_config) as file:
        avp_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.config_vicon, 'r') as file:
        vicon_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.config_vicon_table_pick, 'r') as file:
        vicon_config_table_pick = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.config_vicon_table_drop, 'r') as file:
        vicon_config_table_drop = yaml.load(file, Loader=yaml.FullLoader)

    policy = DecoupledLocomotionManipulationForcePolicy(config=config, 
                                                        avp_config=avp_config,
                                                        vicon_config=vicon_config,
                                                        vicon_config_table_pick =vicon_config_table_pick,
                                                        vicon_config_table_drop =vicon_config_table_drop,
                                                        model_path=args.model_path, 
                                                        use_jit=args.use_jit,
                                                        rl_rate=50, 
                                                        decimation=4)
    policy.run()
