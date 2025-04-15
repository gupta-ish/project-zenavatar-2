import numpy as np
import matplotlib.pyplot as plt

pose_data = np.loadtxt("pose_output_1.txt", delimiter=",")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pose_data[:, 0], pose_data[:, 1], pose_data[:, 2], marker='o', linestyle='-')

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("ArUco Marker Trajectory")

plt.show()
