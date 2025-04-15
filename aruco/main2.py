import pyrealsense2 as rs
import numpy as np
import cv2

# ArUco Dictionary & Detector Parameters
# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# Camera Calibration Parameters 
CAMERA_MATRIX = np.array([[1400, 0, 640], [0, 1400, 360], [0, 0, 1]])  # Some example values
DIST_COEFFS = np.zeros((5, 1))  # Assuming no lens distortion

# From realsense viewer
CAMERA_MATRIX_MAKERSPACE = np.array([[633.6, 0, 642.219],
                                     [0, 633.6, 353.279],
                                     [0,0,1]])


# Expected marker IDs for the tote corners
TOTE_MARKER_IDS = [0, 1, 2, 3]

def compute_grasp_points(corners, ids):
    """
    Compute grasp points using the four detected ArUco markers.
    """
    marker_positions = {}

    # Extract the center position of each marker
    for i in range(len(ids)):
        if ids[i][0] in TOTE_MARKER_IDS:
            marker_positions[ids[i][0]] = np.mean(corners[i].reshape(4, 2), axis=0)

    # Ensure all four markers are detected
    if all(k in marker_positions for k in TOTE_MARKER_IDS):
        mid_top = (marker_positions[0] + marker_positions[1]) / 2
        mid_bottom = (marker_positions[2] + marker_positions[3]) / 2
        mid_left = (marker_positions[0] + marker_positions[2]) / 2
        mid_right = (marker_positions[1] + marker_positions[3]) / 2
        tote_center = (mid_top + mid_bottom) / 2

        grasp_points = {
            "mid_top": mid_top,
            "mid_bottom": mid_bottom,
            "mid_left": mid_left,
            "mid_right": mid_right,
            "tote_center": tote_center
        }

        return grasp_points
    else:
        return None  # If any of the four markers are missing

def detect_totes(frame, depth_frame):
    """
    Detects totes using ArUco markers in the given frame and computes grasp points.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    if ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Compute grasp points from the four corner markers
        grasp_points = compute_grasp_points(corners, ids)

        if grasp_points:
            for key, point in grasp_points.items():
                x, y = int(point[0]), int(point[1])
                depth = depth_frame.get_distance(x, y)  # Get depth value from RealSense
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"{key} ({depth:.2f}m)", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Estimate pose of each marker
        for i in range(len(ids)):
            if ids[i][0] in TOTE_MARKER_IDS:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], markerLength=0.1, cameraMatrix=CAMERA_MATRIX_MAKERSPACE, distCoeffs=DIST_COEFFS
                )
                
                # Draw the marker axes
                cv2.drawFrameAxes(frame, CAMERA_MATRIX_MAKERSPACE, DIST_COEFFS, rvec, tvec, 0.05)

                # Draw bounding boxes around markers
                marker_corners = corners[i].reshape(4, 2)
                x_min, y_min = np.min(marker_corners, axis=0)
                x_max, y_max = np.max(marker_corners, axis=0)
                
                cv2.rectangle(frame, (int(x_min) - 10, int(y_min) - 10), (int(x_max) + 10, int(y_max) + 10), (0, 255, 0), 2)

                # Print 3D position of each marker
                print(f"Marker {ids[i][0]} Position: X={tvec[0][0][0]:.2f}m, Y={tvec[0][0][1]:.2f}m, Z={tvec[0][0][2]:.2f}m")
    else:
        print("None")
    return frame

def main():
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())

            # Detect tote and compute grasp points
            processed_frame = detect_totes(frame, depth_frame)

            # Show output
            cv2.imshow("Tote Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
