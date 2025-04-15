import cv2
import numpy as np
import os

def generate_and_save_aruco_markers(marker_ids, marker_size, output_dir, show=False):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    os.makedirs(output_dir, exist_ok=True)

    for marker_id in marker_ids:
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        filename = os.path.join(output_dir, f"aruco_marker_{marker_id}.png")
        cv2.imwrite(filename, marker_img)

        if show:
            cv2.imshow(f"Marker ID {marker_id}", marker_img)
            cv2.waitKey(500)

    if show:
        cv2.destroyAllWindows()
    print(f"[INFO] {len(marker_ids)} markers saved in '{output_dir}'.")


if __name__ == "__main__":
    MARKER_IDS = [0, 1, 2, 3] 
    MARKER_SIZE = 300          
    OUTPUT_DIR = "aruco_markers"
    DISPLAY_MARKERS = False    
    generate_and_save_aruco_markers(MARKER_IDS, MARKER_SIZE, OUTPUT_DIR, show=DISPLAY_MARKERS)
