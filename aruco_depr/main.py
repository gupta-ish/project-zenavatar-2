import pyrealsense2 as rs
import numpy as np
import cv2

# Load the ArUco dictionary and detector parameters
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# Camera Calibration Parameters (replace with actual calibration values)
CAMERA_MATRIX = np.array([[1400, 0, 640], [0, 1400, 360], [0, 0, 1]])  # Example values
DIST_COEFFS = np.zeros((5, 1))  # Assuming no lens distortion

def detect_totes(frame):
    """
    Detects totes using ArUco markers in the given frame.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    if ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of each marker
        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], markerLength=0.1, cameraMatrix=CAMERA_MATRIX, distCoeffs=DIST_COEFFS
            )
            
            # Draw the axis for visualization
            cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.05)

            # Extract marker corners and compute bounding box
            marker_corners = corners[i].reshape(4, 2)
            x_min, y_min = np.min(marker_corners, axis=0)
            x_max, y_max = np.max(marker_corners, axis=0)
            
            # Draw bounding box around tote (assuming tote is near marker)
            cv2.rectangle(frame, (int(x_min) - 20, int(y_min) - 20), (int(x_max) + 20, int(y_max) + 20), (0, 255, 0), 2)

            # Print position in world coordinates
            print(f"Tote Position (Marker {ids[i][0]}): X={tvec[0][0][0]:.2f}m, Y={tvec[0][0][1]:.2f}m, Z={tvec[0][0][2]:.2f}m")

    return frame

def main():
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # Convert RealSense frame to numpy array
            frame = np.asanyarray(color_frame.get_data())

            # Detect totes using ArUco markers
            processed_frame = detect_totes(frame)

            # Show output
            cv2.imshow("Tote Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
