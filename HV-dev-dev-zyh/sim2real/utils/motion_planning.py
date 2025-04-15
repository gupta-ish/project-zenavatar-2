import torch
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
import os

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

import numpy as np
from scipy.spatial.transform import Slerp, Rotation
from pinocchio import SE3, Quaternion


class MotionPlanning:
    def __init__(self):
        print("Current directory:", os.getcwd())
        config_file = load_yaml("curobo.yaml")
        self.num_dofs = 29
        self.urdf_file = config_file["robot_cfg"][
        "urdf_path"
        ] 
        self.base_link = config_file["robot_cfg"]["base_link"]
        self.ee_link = config_file["robot_cfg"]["ee_link"]

        # Initialize cuRobo kinematics model
        tensor_args = TensorDeviceType()  # use CUDA if available
        robot_cfg = RobotConfig.from_basic(self.urdf_file, self.base_link, self.ee_link, tensor_args)
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)
        world_config={}

        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            "g1.yml",
            world_config,  # Add obstacles if needed
            interpolation_dt=0.02,  # 50Hz waypoints
        )
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup() 


        self.joint_names = [
            # Legs
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            # Torso
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            # Left Arm
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            # Right Arm
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]

        # 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
        #       'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    
    def plan_ee_trajectory(self, robot_state_data, goal_pose_list, ee_frame="left_rubber_hand"):
        """Plan end-effector trajectory from current state to goal pose"""
        # Extract current joint positions (7:7+num_dofs)
        left_arm_joint_indices = [
        self.joint_names.index("left_shoulder_pitch_joint"),
        self.joint_names.index("left_shoulder_roll_joint"),
        self.joint_names.index("left_shoulder_yaw_joint"),
        self.joint_names.index("left_elbow_joint"),
        self.joint_names.index("left_wrist_roll_joint"),
        self.joint_names.index("left_wrist_pitch_joint"),
        self.joint_names.index("left_wrist_yaw_joint")
        ]
    
        # Extract and reshape only the left arm joints
        full_qpos = robot_state_data[:, 7:7+29]  # Full 29-DOF state
        left_arm_qpos = full_qpos[:, left_arm_joint_indices]
        current_qpos = torch.tensor(left_arm_qpos, dtype=torch.float32).cuda().view(1, -1)

        # Create start state
        start_state = JointState.from_position(
            current_qpos,
            joint_names=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint", 
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint"
        ]
        )
        
        # Convert goal to Pose object
        goal_pose = Pose.from_list(goal_pose_list)  # [x,y,z,qw,qx,qy,qz]
        
        # Plan motion
        result = self.motion_gen.plan_single(
            start_state,
            goal_pose,
            MotionGenPlanConfig(
                max_attempts=3,
                # enable_opt=True,
                # balance_constraints=True,
                # end_effector_frame=ee_frame
            )
        )

        traj = result.get_interpolated_plan()  # result.interpolation_dt has the dt between timesteps
        print("trajectory->> ", traj)
        # print("Trajectory Generated: ", result.success)
        
        # if not result.success:
        #     raise RuntimeError("Motion planning failed!")

        # # Extract end-effector waypoints
        # ee_waypoints = []
        # for q in result.trajectory.positions:
        #     ee_pose = self.motion_gen.robot_model.compute_forward_kinematics(q, ee_frame)
        #     ee_waypoints.append(ee_pose.translation.cpu().numpy())
            
        # return np.array(ee_waypoints)
        

    def joint_pose(self, robot_state_data):
        self.qpos = robot_state_data[:, 7:7+self.num_dofs]
        # print("qpos", self.qpos)

    def compute_ee_pose(self, qpos=None):
        """Compute end-effector pose for given joint positions"""
        if qpos is None:
            if hasattr(self, 'qpos'):
                qpos = self.qpos
            else:
                raise ValueError("No joint positions provided")
        
        # Convert to tensor if needed and ensure float32 type
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.tensor(qpos, 
                            device=self.kin_model.tensor_args.device,
                            dtype=torch.float32)  # Explicitly set to float32
        else:
            qpos = qpos.to(device=self.kin_model.tensor_args.device,
                        dtype=torch.float32)  # Convert existing tensor
        
        # Ensure proper shape (batch_size x num_dofs)
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
            
        # Compute forward kinematics
        state = self.kin_model.get_state(qpos)
        
        # Get pose for ee_link
        ee_pose = state.ee_pose
        
        # Convert to position and quaternion
        position = ee_pose.position.cpu().numpy()
        quaternion = ee_pose.quaternion.cpu().numpy()

        return position, quaternion
    
        # return {
        #     'position': position,
        #     'orientation': quaternion,
        #     'pose_matrix': ee_pose.get_matrix().cpu().numpy()
        # }

    def generate_waypoints(self, start_pose, end_pose, num_points):
        """
        Generate straight-line waypoints between start and end poses
        
        Args:
            start_pose: [x, y, z, qw, qx, qy, qz]
            end_pose: [x, y, z, qw, qx, qy, qz]
            num_points: Number of waypoints to generate
            
        Returns:
            np.array: [num_points, 7] array of waypoints
        """
        # Convert to numpy arrays
        start_pos = np.array(start_pose[:3])
        end_pos = np.array(end_pose[:3])
        start_quat = np.array(start_pose[3:])
        end_quat = np.array(end_pose[3:])
        
        # Linear position interpolation
        positions = np.linspace(start_pos, end_pos, num_points)
        
        # Spherical quaternion interpolation (SLERP)
        rotations = Rotation.from_quat([start_quat, end_quat])
        slerp = Slerp([0, 1], rotations)
        quats = slerp(np.linspace(0, 1, num_points)).as_quat()
        
        # Combine position and orientation
        waypoints = np.hstack([positions, quats])
        
        return waypoints
    
    def generate_waypoints_SE(self, start_pose: SE3, end_pose: SE3, num_points: int):
        """
        Generate SE3 waypoints interpolating from start to end pose.

        Args:
            start_pose (pin.SE3): Starting pose
            end_pose (pin.SE3): Ending pose
            num_points (int): Number of waypoints to generate

        Returns:
            list of pin.SE3: Interpolated SE3 poses
        """
        # Extract positions
        start_pos = start_pose.translation
        end_pos = end_pose.translation

        # Extract orientations as pin.Quaternions
        start_quat = Quaternion(start_pose.rotation)
        end_quat = Quaternion(end_pose.rotation)

        # Generate interpolation parameters
        alphas = np.linspace(0, 1, num_points)

        waypoints = []
        for alpha in alphas:
            # Interpolate position
            interp_pos = (1 - alpha) * start_pos + alpha * end_pos
            
            # Interpolate orientation using SLERP
            interp_quat = start_quat.slerp(alpha, end_quat)
            interp_rot = interp_quat.toRotationMatrix()
            
            # Create waypoint
            waypoint = SE3(interp_rot, interp_pos)
            waypoints.append(waypoint)

        return waypoints
