import pyrealsense2 as rs
import numpy as np
import cv2


ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# TODO: ishitag

CAMERA_MATRIX = np.array([[1400, 0, 640], [0, 1400, 360], [0, 0, 1]])  # random vals for now
DIST_COEFFS = np.zeros((5, 1)) 


def compute_grasp_points(corners, ids):
    """
    Compute grasp points using the four ArUco markers.
    """
    marker_positions = {}

    for i in range(len(ids)):
        marker_positions[ids[i][0]] = np.mean(corners[i].reshape(4, 2), axis=0)  

    if all(k in marker_positions for k in [0, 1, 2, 3]):
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
        return None  


def detect_totes(frame):
    """
    Detects totes using ArUco markers in the given frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], markerLength=0.1, cameraMatrix=CAMERA_MATRIX, distCoeffs=DIST_COEFFS
            )
            
            cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.05)

            marker_corners = corners[i].reshape(4, 2)
            x_min, y_min = np.min(marker_corners, axis=0)
            x_max, y_max = np.max(marker_corners, axis=0)
            
            cv2.rectangle(frame, (int(x_min) - 60, int(y_min) - 20), (int(x_max) + 60, int(y_max) + 20), (0, 255, 0), 2)  # bounding box
            print(f"Tote Position (Marker {ids[i][0]}): X={tvec[0][0][0]:.2f}m, Y={tvec[0][0][1]:.2f}m, Z={tvec[0][0][2]:.2f}m")

    return frame

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            processed_frame = detect_totes(frame)
            cv2.imshow("Tote Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
