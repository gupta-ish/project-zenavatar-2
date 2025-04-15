import pyrealsense2 as rs
import numpy as np
import cv2


ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

CAMERA_MATRIX = np.array([[1400, 0, 640], [0, 1400, 360], [0, 0, 1]])

# From realsense viewer
CAMERA_MATRIX_MAKERSPACE = np.array([[633.6, 0, 642.219],
                                     [0, 633.6, 353.279],
                                     [0,0,1]])
DIST_COEFFS = np.zeros((5, 1))

TOTE_WIDTH = 0.6   
TOTE_HEIGHT = 0.4  
TOTE_DEPTH = 0.3  

def draw_3d_cuboid(frame, corners, rvec, tvec):
    """
    Draw a 3D cuboid representing the tote using ArUco pose estimation.
    """

    tote_3D_points = np.array([
        [-TOTE_WIDTH/2, -TOTE_HEIGHT/2, 0],  
        [ TOTE_WIDTH/2, -TOTE_HEIGHT/2, 0],  
        [ TOTE_WIDTH/2,  TOTE_HEIGHT/2, 0],  
        [-TOTE_WIDTH/2,  TOTE_HEIGHT/2, 0], 
        [-TOTE_WIDTH/2, -TOTE_HEIGHT/2, -TOTE_DEPTH],  
        [ TOTE_WIDTH/2, -TOTE_HEIGHT/2, -TOTE_DEPTH],  
        [ TOTE_WIDTH/2,  TOTE_HEIGHT/2, -TOTE_DEPTH],  
        [-TOTE_WIDTH/2,  TOTE_HEIGHT/2, -TOTE_DEPTH]   
    ], dtype=np.float32)

    img_points, _ = cv2.projectPoints(tote_3D_points, rvec, tvec, CAMERA_MATRIX_MAKERSPACE, DIST_COEFFS)
    img_points = img_points.reshape(-1, 2).astype(int)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  
        (4, 5), (5, 6), (6, 7), (7, 4),  
        (0, 4), (1, 5), (2, 6), (3, 7)   
    ]

    for start, end in edges:
        cv2.line(frame, tuple(img_points[start]), tuple(img_points[end]), (0, 255, 0), 2)

def detect_totes(frame, depth_frame):
    """
    Detects totes using ArUco markers and draws a 3D cuboid.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], markerLength=0.1, cameraMatrix=CAMERA_MATRIX_MAKERSPACE, distCoeffs=DIST_COEFFS
            )

            cv2.drawFrameAxes(frame, CAMERA_MATRIX_MAKERSPACE, DIST_COEFFS, rvec, tvec, 0.05)

            x, y = int(np.mean(corners[i][0][:, 0])), int(np.mean(corners[i][0][:, 1]))
            depth = depth_frame.get_distance(x, y)
            print(f"Marker {ids[i][0]} Depth: {depth:.2f}m")

            
            draw_3d_cuboid(frame, corners[i], rvec, tvec)

    return frame

def main():
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

            processed_frame = detect_totes(frame, depth_frame)

            cv2.imshow("Tote Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
