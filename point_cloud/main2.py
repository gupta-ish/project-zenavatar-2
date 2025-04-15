import pyrealsense2 as rs
import numpy as np
import open3d as o3d

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

align = rs.align(rs.stream.color)

try:
    for _ in range(10):  
        frames = pipeline.wait_for_frames()
    
    frames = align.process(frames)
    
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        raise RuntimeError("Could not acquire depth or color frames.")

    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    
    color_image = np.asanyarray(color_frame.get_data())

    pc = rs.pointcloud()
    pc.map_to(color_frame)  
    points = pc.calculate(depth_frame)  
    
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(vtx)
    o3d_pc.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)

    o3d.io.write_point_cloud("realsense_pointcloud.ply", o3d_pc)
    print("Point cloud saved as 'realsense_pointcloud.ply'")

finally:
    pipeline.stop()
