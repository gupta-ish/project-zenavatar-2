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
from sim2real.sim_env.loco_manip import LocoManipSimulator, RealTimePlotter
# from humanoidverse.utils.arm_ik.arm_ik import G1_29_ArmIK
from sim2real.utils.unitree_sdk2py_bridge import ElasticBand

from threading import Thread
import matplotlib.pyplot as plt
from sim2real.teleop.image_server.image_server import ImageServer
from multiprocessing import shared_memory

# Sin curve for xfrc
def sin_force(t):
    return 10 * np.sin(2 * np.pi * 0.05 * t)  # Frequency of 0.5 Hz, amplitude of 10 N

class LocoManipTVSimulator(LocoManipSimulator):
    def __init__(self, config):
        super().__init__(config)
        self.init_tv("head_camera")
    
    def init_scene(self):
        super().init_scene()
        # Customize the viewer options
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    def init_tv(self, camera_name):
        cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.logger.info(f"Camera ID: {cam_id}")
        config = {
            # 'fps': 30,
            'mj_model': self.mj_model,
            'mj_data': self.mj_data,
            'head_camera_type': 'mujoco',
            'head_camera_image_shape': [480, 640],  # Head camera resolution
            'head_camera_id_numbers': [cam_id],
            # 'head_camera_type': 'opencv',
            # 'head_camera_image_shape': [480, 680],  # Head camera resolution
            # 'head_camera_id_numbers': [4],
            # 'wrist_camera_type': 'opencv',
            # 'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
            # 'wrist_camera_id_numbers': [2, 4],
        }
        self.server = ImageServer(config, Unit_Test=False)
        self.tv_thread = Thread(target=self.server.send_process)
        self.tv_thread.daemon = True

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
                    if not self.init_shm_estimation:
                        self.shm_estimated_force = shared_memory.SharedMemory(name="estimated_ee_force_shm")
                        self.estimated_ee_force_shm = np.ndarray((1, 3), dtype=np.float64, buffer=self.shm_estimated_force.buf)
                        self.init_shm_estimation = True
                    EEF = self.estimated_ee_force_shm[0].copy()
                    self.EEF_z = EEF[2]
                    # print("EEF_z: ", simulation.EEF_z, " EE_xfrc: ", simulation.EE_xfrc)
                    plotter.update_plot(self.EEF_z, self.EE_xfrc*2)
                    # plotter.update_plot(simulation.EEF_z, simulation.EE_xfrc*2, simulation.GRF_z)
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
    
    simulation = LocoManipTVSimulator(config)
    simulation.sim_thread.start()
    # Start the TV thread
    simulation.tv_thread.start()
    simulation.run_plot()