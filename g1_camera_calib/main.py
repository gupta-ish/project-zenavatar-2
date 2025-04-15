import numpy as np
from urdfpy import URDF
from pytransform3d.transformations import transform_from_pq

urdf_path = "g1_29dof_with_hand.urdf"  
robot = URDF.load(urdf_path)

# Get transformation from parent to child in homogeneous matrix form
def get_transform_matrix(joint):
    origin = joint.origin
    xyz = origin[:3]  
    rpy = origin[3:]  
    R = euler_to_rotation_matrix(rpy)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T

# Convert roll-pitch-yaw (RPY) to rotation matrix
def euler_to_rotation_matrix(rpy):
    roll, pitch, yaw = rpy  # wow python
    Rx = np.array([[1, 0, 0], 
                   [0, np.cos(roll), -np.sin(roll)], 
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], 
                   [0, 1, 0], 
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], 
                   [np.sin(yaw), np.cos(yaw), 0], 
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

# Compute the transformation from base to camera
def compute_camera_to_robot_transform(robot, camera_link="d435_link"):
    T_base_to_cam = np.eye(4)
    current_link = camera_link
    
    while current_link in robot.link_map:
        parent_joint = robot.link_map[current_link].parent_joint
        if parent_joint is None:
            break  
        
        T_joint = get_transform_matrix(parent_joint)
        T_base_to_cam = T_joint @ T_base_to_cam  # Accumulate transformations
        current_link = parent_joint.parent
        
    return T_base_to_cam

# Compute the final transformation matrix
T_robot_to_camera = compute_camera_to_robot_transform(robot, camera_link="d435_link")
print("Final Transformation Matrix from Robot Base to RGBD Camera:")
print(T_robot_to_camera)

# Example: Converting a pose from the camera frame to the robot base frame
def transform_pose_to_robot_frame(pose_camera, T_robot_to_camera):
    pose_camera_homogeneous = np.hstack((pose_camera[:3], [1]))  
    pose_robot_homogeneous = np.linalg.inv(T_robot_to_camera) @ pose_camera_homogeneous
    return pose_robot_homogeneous[:3]  

# Transforming a sample pose from RGBD camera frame
pose_camera = np.array([0.5, 0.3, 1.2])  # Example camera pose
pose_robot = transform_pose_to_robot_frame(pose_camera, T_robot_to_camera)
print("Pose in Robot Frame:", pose_robot)
