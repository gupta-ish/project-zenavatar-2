import pyrealsense2 as rs
import numpy as np
import cv2
import os

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_LENGTH = 0.1  # meters
OUTPUT_FILE = "pose_output_test.txt"

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters_create()  # Fixed this line
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                          [0, intr.fy, intr.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(intr.coeffs)

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

try:
    while True:
        # Capture frames from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco_detector.detectMarkers(gray)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            for i in range(len(ids)):
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                with open(OUTPUT_FILE, "a") as f:
                    x, y, z = tvecs[i].flatten()
                    f.write(f"{x},{y},{z}\n")
        cv2.imshow("ArUco Detection", color_image)
        
        key = cv2.waitKey(1)
        if key == 27:  # Esc key
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\nPose data is saved to '{OUTPUT_FILE}'")
