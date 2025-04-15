import cv2

def list_available_cameras(max_cameras=10):
    available_cameras = []
    for cam_id in range(max_cameras):
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            available_cameras.append(cam_id)
            cap.release()
    return available_cameras

cameras = list_available_cameras()
print(f"Available camera IDs: {cameras}")
