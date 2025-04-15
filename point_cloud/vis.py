import open3d as o3d

# Load the PLY file
pcd = o3d.io.read_point_cloud("realsense_pointcloud.ply")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd], window_name="RealSense Point Cloud")