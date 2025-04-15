import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

config.enable_record_to_file("realsense_recording.bag")

pipeline.start(config)
print("Recording started... Press Ctrl+C to stop.")

try:
    while True:
        frames = pipeline.wait_for_frames()
except KeyboardInterrupt:
    print("\nRecording stopped.")

finally:
    pipeline.stop()
    print("Saved to realsense_recording.bag")
