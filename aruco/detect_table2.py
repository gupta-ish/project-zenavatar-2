import pyrealsense2 as rs
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_LENGTH = 0.1  # meters
OUTPUT_FILE = "pose_output_1.txt"

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters_create()
# aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                          [0, intr.fy, intr.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(intr.coeffs)

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('ArUco Marker Pose')
trajectory = []  

T_torso_camera_withshiftedy = np.array([[0.0000, -0.743, 0.6691, 0.0],
                                        [-1.000, 0.0000, 0.0000, 0.047],
                                        [0.0000, -0.669, -0.743, 0.462],
                                        [0.0000, 0.000, 0.0000, 1.0000]]) ## this is the transformation matrix from camera to torso in default position

## camera to torso in wide fov rotated position TODO

T_torso_camera_rotated_aruco_wala = np.array([[0.0000, 0, 1, -0.05],
                                        [1.000, 0.0000, 0.0000, 0.0],
                                        [0.0000, -1, -0, 0.462],
                                        [0.0000, 0.000, 0.0000, 1.0000]])

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # corners, ids, _ = aruco_detector.detectMarkers(gray)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
            tvecs_homo =np.append(tvecs, 1.0).reshape(4, 1)
            # breakpoint()
            object_center_in_torso = T_torso_camera_rotated_aruco_wala @ tvecs_homo
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            for i in range(len(ids)):
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                tvec = object_center_in_torso[:3].flatten() # this will work only if one aruco is in frame
                trajectory.append(tvec)

                with open(OUTPUT_FILE, "a") as f:
                    f.write(f"ID: {ids[i][0]}\n")
                    f.write(f"Translation (tvec): {tvec.tolist()}\n")
                    f.write(f"Rotation (rvec): {rvecs[i].flatten().tolist()}\n\n")

            # Update 3D Plot
            ax.clear()
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            ax.set_title('ArUco Marker Pose (Live)')
            if trajectory:
                traj_np = np.array(trajectory)
                ax.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2], marker='o', linestyle='-', color='b')
                ax.scatter(traj_np[-1, 0], traj_np[-1, 1], traj_np[-1, 2], color='r', s=50, label='Current Pose')
                ax.legend()
            plt.draw()
            plt.pause(0.001)

        cv2.imshow("ArUco Detection", color_image)
        key = cv2.waitKey(1)
        if key == 27: 
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()
    print(f"\nPose data is saved to '{OUTPUT_FILE}'")
