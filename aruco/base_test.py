# import pyrealsense2 as rs

# ctx = rs.context()
# devices = ctx.query_devices()
# if len(devices) == 0:
#     print("No RealSense device connected.")
# else:
#     for device in devices:
#         print(f"Device found: {device.get_info(rs.camera_info.name)}")


import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    raise RuntimeError("No RealSense device connected.")

for device in devices:
    print(f"Device found: {device.get_info(rs.camera_info.name)}")
    
    device.hardware_reset()
    print("Hardware reset done. Reconnect the camera and try again.")
