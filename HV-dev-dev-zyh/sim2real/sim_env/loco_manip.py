import mujoco
import mujoco.viewer
import threading
import numpy as np
import time
from loguru import logger
import argparse
import yaml
from threading import Thread

import sys
sys.path.append('../')
sys.path.append('./sim2real')
from sim2real.utils.util import quaternion_to_rotation_matrix, quat_wxyz_to_xyzw, skew_symmetric

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# from std_msgs.msg import Float64MultiArray
from sim2real.sim_env.base_sim import BaseSimulator
# from humanoidverse.utils.arm_ik.arm_ik import G1_29_ArmIK
from sim2real.utils.unitree_sdk2py_bridge import ElasticBand

from threading import Thread
import matplotlib.pyplot as plt
from multiprocessing import shared_memory
from scipy.spatial.transform import Rotation as R

class RealTimePlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.eef_force_line, = self.ax.plot([], [], label='EEF Force', color='blue')
        self.xfrc_sum_line, = self.ax.plot([], [], label='xfrc Sum', color='orange')
        self.grf_adjusted_line, = self.ax.plot([], [], label='Adjusted GRF', color='green') 

        self.time_data = []
        self.eef_data = []
        self.xfrc_sum_data = []
        self.grf_adjusted_data = [] 
        self.start_time = time.time()
        self.running = True

        # Configure the plot
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-50, 50)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Force (N)')
        self.ax.legend(loc='upper right')
        self.ax.grid()

    def update_plot(self, eef_force, xfrc_sum, grf_adjusted):
        """Add new data points to the plot."""
        current_time = time.time() - self.start_time
        self.time_data.append(current_time)
        self.eef_data.append(eef_force)
        self.xfrc_sum_data.append(xfrc_sum)
        self.grf_adjusted_data.append(grf_adjusted) 

        # Update the plot range if needed
        if current_time > self.ax.get_xlim()[1]:
            self.ax.set_xlim(0, current_time + 1)

        # Update data
        self.eef_force_line.set_data(self.time_data, self.eef_data)
        self.xfrc_sum_line.set_data(self.time_data, self.xfrc_sum_data)
        self.grf_adjusted_line.set_data(self.time_data, self.grf_adjusted_data)  # 更新 Adjusted GRF 曲线

        # Redraw the plot
        self.ax.figure.canvas.draw()
        self.ax.figure.canvas.flush_events()
    
    def update_plot(self, eef_force, xfrc_sum):
        """Add new data points to the plot."""
        current_time = time.time() - self.start_time
        self.time_data.append(current_time)
        self.eef_data.append(eef_force)
        self.xfrc_sum_data.append(xfrc_sum)

        # Update the plot range if needed
        if current_time > self.ax.get_xlim()[1]:
            self.ax.set_xlim(0, current_time + 1)

        # Update data
        self.eef_force_line.set_data(self.time_data, self.eef_data)
        self.xfrc_sum_line.set_data(self.time_data, self.xfrc_sum_data)

        # Redraw the plot
        self.ax.figure.canvas.draw()
        self.ax.figure.canvas.flush_events()

    def stop(self):
        self.running = False

# Sin curve for xfrc
def sin_force(t):
    return 20 * np.sin(2 * np.pi * 0.025 * t)

class LocoManipSimulator(BaseSimulator):
    def __init__(self, config):
        super().__init__(config)
        self.init_shm_estimation=False
        self.EE_xfrc = 0
        self.EEF_z = 0
        self.GRF_z = 0
        self.t = 0
        self.dt = 0.01  # 10 ms time step
        self.pose_shm = shared_memory.SharedMemory(name="pose_of_G1", create=True, size=6 * 8)
        self.pose_array = np.ndarray((6,), dtype=np.float64, buffer=self.pose_shm.buf)

    def init_scene(self):
        super().init_scene()
        NUM_FEET_SENSORS = 8
        # Assuming you know the order of the sensors in the XML
        self.ffss_idx = len(self.mj_data.sensordata) - NUM_FEET_SENSORS * 3
        # Customize the viewer options
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    
    def sim_step(self):
        self.unitree_bridge.PublishLowState()
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
                    self.mj_data.qpos[:3], self.mj_data.qvel[:3]
                )
        
        # Yuanhang: Testing the estimated force
        if self.elastic_band.estimate:
            self.EE_xfrc = -abs(sin_force(self.t))
            self.EE_xfrc = 30
            self.mj_data.xfrc_applied[self.mj_model.body("left_rubber_hand").id, 0] = self.EE_xfrc
            self.mj_data.xfrc_applied[self.mj_model.body("right_rubber_hand").id, 0] = self.EE_xfrc
            # self.mj_data.xfrc_applied[self.mj_model.body("left_wrist_yaw_link").id, 2] = self.EE_xfrc
            # self.mj_data.xfrc_applied[self.mj_model.body("right_wrist_yaw_link").id, 2] = self.EE_xfrc
            self.t += self.dt

        # Yuanhang: Hardcode for testing
        # self.EE_xfrc = -20
        # self.mj_data.xfrc_applied[self.mj_model.body("left_wrist_yaw_link").id, 2] = -20
        # self.mj_data.xfrc_applied[self.mj_model.body("right_wrist_yaw_link").id, 2] = -20
        
        self.compute_torques()
        if self.unitree_bridge.free_base:
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else: self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.get_robot_pose()
        
    def get_robot_pose(self):
        pos = self.mj_data.xpos[self.base_id]
        rotmat = self.mj_data.xmat[self.base_id].reshape(3, 3)
        euler = R.from_matrix(rotmat).as_euler('xyz', degrees=True)

        self.pose_array[:3] = pos
        self.pose_array[3:] = euler
    
    def get_robot_states(self):
        # Get the low state of the robot
        raise NotImplementedError
    
    def get_feet_forces(self):
        # Get the force in feet (w.r.t. the feet frame)
        forces_feet = {'left': np.zeros((4,3)),
                       'right': np.zeros((4,3))}
        for i in range(4):
            forces_feet['left'][i] = -self.mj_data.sensordata[self.ffss_idx+i*3:self.ffss_idx+i*3+3]
            forces_feet['right'][i] = -self.mj_data.sensordata[self.ffss_idx+(i+4)*3:self.ffss_idx+(i+4)*3+3]
        # print("forces_feet[left][0]: ", forces_feet['left'][0])
        return forces_feet

    def update_site(self, info=None):
        # Visualize COM
        self.mj_model.site_pos[self.mj_model.site("com_marker").id] = self.force_estimator.robot.data.com[0]
        # Visualize the feet and hands
        self.mj_model.site_pos[self.mj_model.site("left_foot_marker").id] = self.force_estimator.robot.data.oMf[self.force_estimator.frames_idx['left_ankle_roll_link']].translation
        self.mj_model.site_pos[self.mj_model.site("right_foot_marker").id] = self.force_estimator.robot.data.oMf[self.force_estimator.frames_idx['right_ankle_roll_link']].translation
        self.mj_model.site_pos[self.mj_model.site("left_hand_marker").id] = self.force_estimator.robot.data.oMf[self.force_estimator.frames_idx['left_rubber_hand']].translation
        self.mj_model.site_pos[self.mj_model.site("right_hand_marker").id] = self.force_estimator.robot.data.oMf[self.force_estimator.frames_idx['right_rubber_hand']].translation
        # Extract positions from info
        if info is not None:
            com_pos, p_1, p_2, p_3, p_4 = info
            # Update site positions
            self.mj_model.site_pos[self.mj_model.site("com_marker").id] = com_pos
            self.mj_model.site_pos[self.mj_model.site("p1_marker").id] = p_1
            self.mj_model.site_pos[self.mj_model.site("p2_marker").id] = p_2
            self.mj_model.site_pos[self.mj_model.site("p3_marker").id] = p_3
            self.mj_model.site_pos[self.mj_model.site("p4_marker").id] = p_4
    
    def run_plot(self):
        # Real-time plotter (only works on Linux, NOT on Mac)
        if sys.platform == 'linux':
            t = 0
            dt = 0.05  # 10 ms time step
            plotter = RealTimePlotter()
            plt.ion()
            plt.show()
            while plotter.running:
                if self.elastic_band.estimate:
                    # if not self.init_shm_estimation:
                    #     self.shm_estimated_left_ee_force = shared_memory.SharedMemory(name="estimated_left_ee_force_shm")
                    #     self.estimated_left_ee_force_shm = np.ndarray((1, 3), dtype=np.float64, buffer=self.shm_estimated_left_ee_force.buf)
                    #     self.shm_estimated_right_ee_force = shared_memory.SharedMemory(name="estimated_right_ee_force_shm")
                    #     self.estimated_right_ee_force_shm = np.ndarray((1, 3), dtype=np.float64, buffer=self.shm_estimated_right_ee_force.buf)
                    #     self.init_shm_estimation = True
                    # Left_EEF = self.estimated_left_ee_force_shm[0].copy()
                    # Right_EEF = self.estimated_right_ee_force_shm[0].copy()
                    # self.left_EEF_z = Left_EEF[2]
                    # self.right_EEF_z = Right_EEF[2]
                    # self.EEF_z = self.left_EEF_z + self.right_EEF_z
                    # # print("EEF_z: ", simulation.EEF_z, " EE_xfrc: ", simulation.EE_xfrc)
                    # plotter.update_plot(self.EEF_z, self.EE_xfrc*2)
                    # plotter.update_plot(simulation.EEF_z, simulation.EE_xfrc*2, simulation.GRF_z)
                    pass
                time.sleep(dt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/g1_29dof_free.yaml', help='config file')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config.get("INTERFACE", None):
        if sys.platform == "linux":
            config["INTERFACE"] = "lo"
        elif sys.platform == "darwin":
            config["INTERFACE"] = "lo0"
        else: raise NotImplementedError("Only support Linux and MacOS.")
        ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    else:
        ChannelFactoryInitialize(config["DOMAIN_ID"])
    
    simulation = LocoManipSimulator(config)
    simulation.sim_thread.start()
    simulation.run_plot()
