import cv2
import os

# Create output folder
os.makedirs("g1_output_images", exist_ok=True)

print("Starting camera stream...")
# Open default camera (change to 1 or 2 if needed)
cap = cv2.VideoCapture(0)  # 2 is webcam, 1 connecting to phone camera
print("Some Camera opened.")
# Check if camera is opened
if not cap.isOpened():
    print("❌ Camera not found or cannot be accessed.")
    exit()

print("✅ Camera connected. Press 'q' to quit.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    # Save the frame as an image
    filename = f"g1_output_images/frame_{frame_count:04d}.png"
    cv2.imwrite(filename, frame)
    print(f"Saved {filename}")
    frame_count += 1

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Camera stream ended.")
