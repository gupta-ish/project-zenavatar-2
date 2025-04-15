import cv2
import numpy as np

# Load ArUco dictionary
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Define marker ID and size
MARKER_ID = 0
MARKER_SIZE = 300  # Pixels

# Generate marker image
marker_img = cv2.aruco.generateImageMarker(ARUCO_DICT, MARKER_ID, MARKER_SIZE)

# Save the marker image
cv2.imwrite("aruco_marker.png", marker_img)

# Display the marker
cv2.imshow("ArUco Marker", marker_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
