import numpy as np
from scipy.spatial.transform import Rotation as R

# camera position in torso frame
translation = np.array([0.06, 0.0, 0.45])

# rpy
euler_xyz = [0, -0.8, -1.57]  

rotation_matrix = R.from_euler('xyz', euler_xyz).as_matrix()

T_torso_from_camera = np.eye(4)
T_torso_from_camera[:3, :3] = rotation_matrix
T_torso_from_camera[:3, 3] = translation

print("Transformation matrix (Torso <-- Camera):")
print(T_torso_from_camera)

point_camera = np.array([0.1, 0.0, 0.0, 1])  # homogeneous coordinates [I have put this just as an example, enter actual points here]


point_torso = T_torso_from_camera @ point_camera

print("Point in Camera Frame:")
print(point_camera[:3])
print("Point in Torso Frame:")
print(point_torso[:3])  
