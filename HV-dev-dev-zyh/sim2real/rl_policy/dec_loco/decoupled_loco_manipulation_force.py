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
from decoupled_loco_manipulation import DecoupledLocomotionManipulationPolicy
from multiprocessing import shared_memory, Array, Lock
from sim2real.teleop.image_server.image_client import ImageClient
from sim2real.teleop.open_television.tv_wrapper import TeleVisionWrapper

from utils.robot_arm_ik import G1_29_ArmIK
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin                             
import time
from pinocchio import casadi as cpin                
from pinocchio.visualize import MeshcatVisualizer
from sim2real.utils.util import normalize, unnormalize
import mujoco

from sim2real.utils.util import quat_rotate_inverse_numpy, quat_rotate_numpy, quat_wxyz_to_xyzw

def clock_input():
    t = time.time()
    t -= int(t / 1000) * 1000

    frequency = 1.5
    phase = 0.5

    gait_indices = t * frequency - int(t * frequency)
    foot_indices = np.array([phase, 0]) + gait_indices

    clock_inputs = np.sin(2 * np.pi * foot_indices)

    return clock_inputs[None, :]

class DecoupledLocomotionManipulationForcePolicy(DecoupledLocomotionManipulationPolicy):
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
        self.apply_force = np.array([[0]])
        self.left_ee_force = np.zeros((1, 3))
        self.right_ee_force = np.zeros((1, 3))
        self.init_force_shm = False
        self.estimating_force = False
        self.ref_upper_dof_pos *= 0.0
        
        self.left_ee_comp = np.zeros((1, self.num_upper_dofs))
        self.right_ee_comp = np.zeros((1, self.num_upper_dofs))
        
        self.left_ee_force_estimator_output = np.zeros((1, 3))
        self.right_ee_force_estimator_output = np.zeros((1, 3))

        self.force_min_x, self.force_max_x = self.config["force_range"]["apply_force_x_range"][0], \
                                             self.config["force_range"]["apply_force_x_range"][1]
        self.force_min_y, self.force_max_y = self.config["force_range"]["apply_force_y_range"][0], \
                                             self.config["force_range"]["apply_force_y_range"][1]
        self.force_min_z, self.force_max_z = self.config["force_range"]["apply_force_z_range"][0], \
                                             self.config["force_range"]["apply_force_z_range"][1]
        self.force_min = np.array([self.force_min_x, self.force_min_y, self.force_min_z])
        self.force_max = np.array([self.force_max_x, self.force_max_y, self.force_max_z])

        # Initialize Pinocchio model
        self.init_robot_model()

    # def init_robot_model(self):
    #     # Initialize the robot model using Pinocchio
    #     self.robot = pin.RobotWrapper.BuildFromURDF(
    #         self.config["ASSET_ROOT"] + '/' + self.config["ASSET_FILE"],
    #         self.config["ASSET_ROOT"]
    #     )
    #     # List all the frames
    #     self.frames_idx = {}
    #     for i, frame in enumerate(self.robot.model.frames):
    #         print(f"Frame {i}: Name = {frame.name}, Type = {frame.type}")
    #         self.frames_idx[frame.name] = i
    #     self.robot_mass = pin.computeTotalMass(self.robot.model, self.robot.data)
    #     self.num_dofs = self.robot.model.nq - 7
    #     self.vis_pin = self.config["VISUALIZE_PIN"]
    #     if self.vis_pin:
    #         # Initialize the Meshcat visualizer for visualization
    #         self.vis = MeshcatVisualizer(self.robot.model, self.robot.collision_model, self.robot.visual_model)
    #         self.vis.initViewer(open=True) 
    #         self.vis.loadViewerModel("pinocchio") 
    #         self.vis.display(pin.neutral(self.robot.model))'

    def init_robot_model(self):
        # Load Mujoco model
        self.model = mujoco.MjModel.from_xml_path(self.config["ROBOT_SCENE"])
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        # Initialize body indices
        self.frames_idx = {
            'left_rubber_hand': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_rubber_hand'),
            'right_rubber_hand': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_rubber_hand'),
            'left_wrist_yaw_link': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_wrist_yaw_link'),
            'right_wrist_yaw_link': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist_yaw_link')
        }
        
        self.robot_mass = np.sum(self.model.body_mass)
        self.num_dofs = self.model.nv - 6  # Excluding floating base DOFs if applicable

    def setup_policy(self, model_path, use_jit):
        # Load the ONNX model
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        
        # Retrieve all input and output names from the ONNX model
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]
        
        def policy_act(obs_dict):
            """
            Perform inference using the ONNX policy model.
            
            Args:
                obs_dict (dict): A dictionary containing:
                    - "actor_obs": Observation input for the actor.
                    - "estimator_obs": Observation input for the force estimators.
            
            Returns:
                dict: A dictionary containing:
                    - "action": The action predicted by the actor.
                    - "left_ee_force_estimator_output": Estimated left end-effector force.
                    - "right_ee_force_estimator_output": Estimated right end-effector force.
            """
            # Prepare the input dictionary for ONNX inference
            input_feed = {
                "actor_obs": obs_dict["actor_obs"],
                "estimator_obs": obs_dict["estimator_obs"]
            }
            
            # Run ONNX inference and obtain all outputs
            outputs = self.onnx_policy_session.run(self.onnx_output_names, input_feed)
            
            # Return results as a dictionary
            return {
                "action": outputs[0], 
                "left_ee_force_estimator_output": outputs[1], 
                "right_ee_force_estimator_output": outputs[2]
            }

        # Assign the policy function to self.policy for later use
        self.policy = policy_act
    
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

        # Yuanhang: get the estimated force here
        if self.init_force_shm and self.estimating_force: 
            estimated_ee_force = self.estimated_ee_force_shm.copy()
            print("estimated_ee_force: ", estimated_ee_force)

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
                                    cos_phase,
                                    dof_pos_minus_default, 
                                    dof_vel*0.05,
                                    history_actor,
                                    # phase_time,
                                    projected_gravity,
                                    self.ref_upper_dof_pos,
                                    sin_phase,
                                    ], axis=1)
            estimator_obs = np.concatenate([self.last_policy_action, 
                                    base_ang_vel*0.25, 
                                    self.ang_vel_command, 
                                    self.base_height_command*2.0,
                                    self.lin_vel_command,
                                    self.stand_command,
                                    cos_phase,
                                    dof_pos_minus_default, 
                                    dof_vel*0.05,
                                    history_estimator,
                                    self.left_ee_comp*0.1,
                                    # phase_time,
                                    projected_gravity,
                                    self.ref_upper_dof_pos,
                                    self.right_ee_comp*0.1,
                                    sin_phase
                                    ], axis=1)
        else:
            actor_obs = np.concatenate([self.last_policy_action, 
                                # self.apply_force*0.1,
                                base_ang_vel*0.25, 
                                self.ang_vel_command, 
                                self.base_height_command*2.0,
                                self.lin_vel_command, 
                                self.stand_command,
                                cos_phase,
                                dof_pos_minus_default, 
                                dof_vel*0.05,
                                # phase_time,
                                projected_gravity,
                                self.ref_upper_dof_pos,
                                sin_phase,
                                ], axis=1)
            estimator_obs = np.concatenate([self.last_policy_action,
                                # self.apply_force*0.1,
                                base_ang_vel*0.25, 
                                self.ang_vel_command, 
                                self.base_height_command*2.0,
                                self.lin_vel_command, 
                                self.stand_command,
                                cos_phase,
                                dof_pos_minus_default, 
                                dof_vel*0.05,
                                self.left_ee_comp*0.1,
                                # phase_time,
                                projected_gravity,
                                self.ref_upper_dof_pos,
                                self.right_ee_comp*0.1,
                                sin_phase,
                                ], axis=1)
        if self.history_handler:
            self.history_handler.add("apply_force", self.apply_force*self.obs_scales["apply_force"])
            self.history_handler.add("base_ang_vel", base_ang_vel*self.obs_scales["base_ang_vel"])
            self.history_handler.add("command_lin_vel", self.lin_vel_command*self.obs_scales["command_lin_vel"])
            self.history_handler.add("command_ang_vel", self.ang_vel_command*self.obs_scales["command_ang_vel"])
            self.history_handler.add("command_stand", self.stand_command*self.obs_scales["command_stand"])
            self.history_handler.add("command_base_height", self.base_height_command*self.obs_scales["command_base_height"])
            self.history_handler.add("dof_pos", dof_pos_minus_default*self.obs_scales["dof_pos"])
            self.history_handler.add("dof_vel", dof_vel*self.obs_scales["dof_vel"])
            self.history_handler.add("projected_gravity", projected_gravity*self.obs_scales["projected_gravity"])
            self.history_handler.add("ref_upper_dof_pos", self.ref_upper_dof_pos*self.obs_scales["ref_upper_dof_pos"])
            self.history_handler.add("actions", self.last_policy_action*self.obs_scales["actions"])
            # self.history_handler.add("phase_time", phase_time*self.obs_scales["phase_time"])
            self.history_handler.add("sin_phase", sin_phase*self.obs_scales["sin_phase"])
            self.history_handler.add("cos_phase", cos_phase*self.obs_scales["cos_phase"])
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
    
    def rl_inference(self, robot_state_data):
        obs_dict = self.prepare_obs_for_rl(robot_state_data)
        policy_output_dict = self.policy(obs_dict)
        policy_action = policy_output_dict["action"]
        left_ee_force_estimator_output = policy_output_dict["left_ee_force_estimator_output"]
        right_ee_force_estimator_output = policy_output_dict["right_ee_force_estimator_output"]
        policy_action = np.clip(policy_action, -100, 100)
        
        # Lower body actions
        self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale
        # Combine upper body actions
        scaled_policy_action = np.concatenate([scaled_policy_action, self.ref_upper_dof_pos], axis=1)

        return scaled_policy_action, left_ee_force_estimator_output, right_ee_force_estimator_output
    
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
           # Control upper qpos
           upper_body_qpos, upper_body_tauff = self.upper_body_controller.get_q_tau(
               self.waypoints_left[self.waypoint_index],
               self.waypoints_right[self.waypoint_index]
           )
           self.ref_upper_dof_pos[:, -self.num_upper_dofs:] = upper_body_qpos
           # Control upper tau
        #    cmd_tau[-14:] = upper_body_tauff
        
        # Get policy action and estimated force
        scaled_policy_action, left_ee_force_estimator_output, right_ee_force_estimator_output = self.rl_inference(self.robot_state_data)
        # Unnormalize the estimated force and rotate to global frame
        left_ee_estimated_force_unnormalized = unnormalize(left_ee_force_estimator_output, self.force_min, self.force_max)
        right_ee_estimated_force_unnormalized = unnormalize(right_ee_force_estimator_output, self.force_min, self.force_max)
        global_left_ee_estimated_force = quat_rotate_numpy(self.robot_state_data[:, 3:7], left_ee_estimated_force_unnormalized)
        global_right_ee_estimated_force = quat_rotate_numpy(self.robot_state_data[:, 3:7], right_ee_estimated_force_unnormalized)
        if self.init_force_shm: 
            self.estimated_left_ee_force_shm[0] = global_left_ee_estimated_force.copy()
            self.estimated_right_ee_force_shm[0] = global_right_ee_estimated_force.copy()
        
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
        # TODO [Yuanhang]: 
        # 1. add tau compensation
        # 2. update tau compensation based on the estimated force
        qpos = self.robot_state_data[0, :7+self.num_dofs]
        qvel = self.robot_state_data[0, 7+self.num_dofs:7+self.num_dofs+6+self.num_dofs]
        qpos[3:7] = quat_wxyz_to_xyzw(qpos[3:7])
        if self.init_force_shm:
            self.left_ee_comp[0], self.right_ee_comp[0] = self.calculate_ee_torque_compensation(global_left_ee_estimated_force, 
                                                                                global_right_ee_estimated_force, 
                                                                                qpos, qvel)
        cmd_tau[-self.num_upper_dofs:] -= (self.left_ee_comp[0] + self.right_ee_comp[0])
        # print("right_ee_comp: ", self.right_ee_comp)
        if self.history_handler:
            self.history_handler.add("left_ee_comp", self.left_ee_comp*self.obs_scales["left_ee_comp"])
            self.history_handler.add("right_ee_comp", self.right_ee_comp*self.obs_scales["right_ee_comp"])
        # Visualize the robot in meshcat
        # if self.vis_pin: self.vis.display(qpos)
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)

    # def calculate_ee_torque_compensation(self, left_ee_force, right_ee_force, qpos, qvel):
    #     print("left_ee_force: ", left_ee_force)
    #     print("right_ee_force: ", right_ee_force)
    #     # Update pinocchio kinematics
    #     pin.forwardKinematics(self.robot.model, self.robot.data, qpos, qvel)
    #     pin.updateFramePlacements(self.robot.model, self.robot.data)
    #     Jl = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['left_rubber_hand'])[0:3, 6:]
    #     Jr = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['right_rubber_hand'])[0:3, 6:]
    #     Jl_upper = Jl[:, -self.num_upper_dofs:]
    #     Jr_upper = Jr[:, -self.num_upper_dofs:]
    #     left_ee_comp = np.dot(Jl_upper.T, left_ee_force[0])
    #     right_ee_comp = np.dot(Jr_upper.T, right_ee_force[0])
    #     return left_ee_comp, right_ee_comp

    def calculate_ee_torque_compensation(self, left_ee_force, right_ee_force, qpos, qvel):
        print("left_ee_force: ", left_ee_force, "right_ee_force: ", right_ee_force)
        # Update Mujoco kinematics
        # mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos.copy()
        # Do not use mj_kinematics, it does more than foward the position kinematics!
        # mujoco.mj_kinematics(model, data)
        mujoco.mj_fwdPosition(self.model, self.data)
        # Compute Jacobians using Mujoco
        Jl = np.zeros((6, self.model.nv))
        Jr = np.zeros((6, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, Jl[:3], Jl[3:], self.frames_idx['left_wrist_yaw_link'])
        mujoco.mj_jacBody(self.model, self.data, Jr[:3], Jr[3:], self.frames_idx['right_wrist_yaw_link'])
        Jl_upper = Jl[:3, -self.num_upper_dofs:]
        Jr_upper = Jr[:3, -self.num_upper_dofs:]
        print("Jl_upper:", Jl_upper[2, :])
        print("Jr_upper:", Jr_upper[2, :])
        left_ee_comp = np.dot(Jl_upper.T, left_ee_force[0])
        right_ee_comp = np.dot(Jr_upper.T, right_ee_force[0])
        return left_ee_comp, right_ee_comp

    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        if keycode == " ":
            self.estimating_force = not self.estimating_force
            self.logger.info(f"Estimating force: {self.estimating_force}")
            if not self.init_force_shm:
                self.init_force_shm = True
                self.shm_estimated_left_ee_force = shared_memory.SharedMemory(name="estimated_left_ee_force_shm", create=True, size=8*3)
                self.shm_estimated_right_ee_force = shared_memory.SharedMemory(name="estimated_right_ee_force_shm", create=True, size=8*3)
                self.estimated_left_ee_force_shm = np.ndarray((1, 3), dtype=np.float64, buffer=self.shm_estimated_left_ee_force.buf)
                self.estimated_right_ee_force_shm = np.ndarray((1, 3), dtype=np.float64, buffer=self.shm_estimated_right_ee_force.buf)
    
    def handle_joystick_button(self, cur_key):
        super().handle_joystick_button(cur_key)
        if cur_key == "L1+R1":
            self.estimating_force = not self.estimating_force
            self.logger.info(f"Estimating force: {self.estimating_force}")
            if not self.init_force_shm:
                self.init_force_shm = True
                self.shm_estimated_left_ee_force = shared_memory.SharedMemory(name="estimated_left_ee_force_shm", create=True, size=8*3)
                self.shm_estimated_right_ee_force = shared_memory.SharedMemory(name="estimated_right_ee_force_shm", create=True, size=8*3)
                self.estimated_left_ee_force_shm = np.ndarray((1, 3), dtype=np.float64, buffer=self.shm_estimated_left_ee_force.buf)
                self.estimated_right_ee_force_shm = np.ndarray((1, 3), dtype=np.float64, buffer=self.shm_estimated_right_ee_force.buf)
    
    def _get_obs_history_actor(self):
        assert "history_config" in self.config.keys()
        assert "history_actor" in self.config["history_config"].keys()
        history_config = self.config["history_config"]["history_actor"]
        history_list = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_array = self.history_handler.query(key)[:, :history_length]
            history_array = history_array.reshape(history_array.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_list.append(history_array)
        return np.concatenate(history_list, axis=1)
    
    def _get_obs_history_estimator(self):
        assert "history_config" in self.config.keys()
        assert "history_estimator" in self.config["history_config"].keys()
        history_config = self.config["history_config"]["history_estimator"]
        history_list = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_array = self.history_handler.query(key)[:, :history_length]
            history_array = history_array.reshape(history_array.shape[0], -1) # Shape: [4096, history_length*obs_dim]
            history_list.append(history_array)
        return np.concatenate(history_list, axis=1)

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