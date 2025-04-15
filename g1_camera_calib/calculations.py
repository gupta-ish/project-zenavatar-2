import numpy as np

def print_matrix(name, matrix):
    print(f"\n{name}:\n" + "\n".join(["\t".join([f"{val: .4f}" for val in row]) for row in matrix]))


T_pelvis_torso = np.array([
    [1, 0, 0, -0.0039635],  # Small translation along X-axis
    [0, 1, 0, 0],           
    [0, 0, 1, 0.054],       # 54mm translation along Z-axis
    [0, 0, 0, 1]            
])

theta = 0.8308 
R_y = np.array([
    [np.cos(theta), 0, np.sin(theta)],  
    [0, 1, 0],                         
    [-np.sin(theta), 0, np.cos(theta)] 
])

t_torso_camera = np.array([0.0576235, 0.01753, 0.41987])  # Given in URDF

# Torso to Camera
T_torso_camera = np.eye(4)  
T_torso_camera[:3, :3] = R_y  
T_torso_camera[:3, 3] = t_torso_camera  
#Pelvis to Camera
T_pelvis_camera = np.dot(T_pelvis_torso, T_torso_camera)
R_pelvis_camera = np.dot(T_pelvis_torso[:3, :3], T_torso_camera[:3, :3])
t_pelvis_camera = np.dot(T_pelvis_torso[:3, :3], t_torso_camera) + T_pelvis_torso[:3, 3]
T_pelvis_camera_explicit = np.eye(4)
T_pelvis_camera_explicit[:3, :3] = R_pelvis_camera
T_pelvis_camera_explicit[:3, 3] = t_pelvis_camera

print("=" * 50)
print("Information Extracted from URDF")
print_matrix("Transformation from Pelvis to Torso (T_pelvis_torso)", T_pelvis_torso)
print_matrix("Rotation Matrix from Torso to Camera (R_y)", R_y)
print("\nTranslation Vector from Torso to Camera")
print("\t".join([f"{val: .4f}" for val in t_torso_camera]))
print_matrix("T_torso_camera", T_torso_camera)
print("\nComputed Rotation and Translation from Pelvis to Camera")
print_matrix("Rotation Component (R_pelvis_camera)", R_pelvis_camera)
print("\nTranslation Component (t_pelvis_camera)")
print("\t".join([f"{val: .4f}" for val in t_pelvis_camera]))
print_matrix("T_pelvis_camera_explicit", T_pelvis_camera_explicit)
print("=" * 50)
