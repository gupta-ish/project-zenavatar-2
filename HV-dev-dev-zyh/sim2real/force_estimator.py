import numpy as np
import time
from loguru import logger
from threading import Thread

import pinocchio as pin              
from pinocchio.visualize import MeshcatVisualizer  

import sys
sys.path.append('../')
sys.path.append('./sim2real')
from sim2real.utils.util import quaternion_to_rotation_matrix, quat_wxyz_to_xyzw, skew_symmetric

from sim2real.utils.robot import Robot
from loop_rate_limiters import RateLimiter
from multiprocessing import shared_memory
import argparse
import yaml
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

class ForceEstimator:
    def __init__(self, robot_config):
        self.config = robot_config
        self.init_robot_model()
        self.rate = RateLimiter(self.config.get("ESTIMATE_FORCE_RATE", 5))
        if config.get("INTERFACE", None):
            ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
        else:
            ChannelFactoryInitialize(config["DOMAIN_ID"])
        num_states_digits = 7 + 6 + 6 + 6 + self.num_dofs*4
        self.shm_state = None
        shm_name = "robot_state_data_shm"
        while self.shm_state is None:
            try:
                self.shm_state = shared_memory.SharedMemory(name=shm_name)
                logger.info(f"Connected to shared memory {shm_name}!")
            except FileNotFoundError:
                print("Shared memory not found. Waiting...")
                time.sleep(1)
        self.robot_state_data_shm = np.ndarray((1, num_states_digits), dtype=np.float64, buffer=self.shm_state.buf)
        self.shm_estimated_force = shared_memory.SharedMemory(name="estimated_ee_force_shm", create=True, size=8*3)
        self.estimated_ee_force_shm = np.ndarray((1, 3), dtype=np.float64, buffer=self.shm_estimated_force.buf)

    def init_robot_model(self):
        # Initialize the robot model using Pinocchio
        self.robot = pin.RobotWrapper.BuildFromURDF(
            self.config["ASSET_ROOT"] + '/' + self.config["ASSET_FILE"],
            self.config["ASSET_ROOT"]
        )
        # List all the frames
        self.frames_idx = {}
        for i, frame in enumerate(self.robot.model.frames):
            print(f"Frame {i}: Name = {frame.name}, Type = {frame.type}")
            self.frames_idx[frame.name] = i
        self.robot_mass = pin.computeTotalMass(self.robot.model, self.robot.data)
        self.num_dofs = self.robot.model.nq - 7
        self.vis_pin = self.config["VISUALIZE_PIN"]
        if self.vis_pin:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(self.robot.model, self.robot.collision_model, self.robot.visual_model)
            self.vis.initViewer(open=True) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.display(pin.neutral(self.robot.model))
        # SRB Parameters
        self.SRB_Init(solver=self.config["SRB_SOLVER"])

    def SRB_Init(self, solver):
        if solver == "SRB_FULL":
            self.A = np.zeros((21, 12))
            self.A[0:3, 0:3] = np.eye(3); self.A[0:3, 3:6] = np.eye(3)
            self.A[0:3, 6:9] = np.eye(3); self.A[0:3, 9:12] = np.eye(3)
            # self.A[6:9, 6:9] = np.eye(3); self.A[6:9, 9:12] = -np.eye(3)
            self.A[8, 8]=1.0 ;self.A[8, 11] = -1.0
            self.b = np.zeros(21)
            self.solve_grf_eef = self.solve_SRB_grf_eef_full
        elif solver == "SRB_SIMPLE":
            self.A = np.zeros((35, 12))
            self.A[0:3, 0:3] = np.eye(3); self.A[0:3, 3:6] = np.eye(3)
            self.A[0:3, 6:9] = np.eye(3); self.A[0:3, 9:12] = np.eye(3)
            self.A[3:6, 6:9] = np.eye(3); self.A[3:6, 9:12] = np.eye(3)
            self.A[5, 8] = 1.0 ;self.A[5, 11] = -1.0
            self.b = np.zeros(35)
            self.solve_grf_eef = self.solve_SRB_grf_eef_simple
        elif solver == "SRB_Z":
            self.A = np.zeros((35, 12))
            self.A[0:3, 0:3] = np.eye(3); self.A[0:3, 3:6] = np.eye(3)
            self.A[0:3, 6:9] = np.eye(3); self.A[0:3, 9:12] = np.eye(3)
            self.A[3:6, 6:9] = np.eye(3); self.A[3:6, 9:12] = np.eye(3)
            self.A[5, 8] = 1.0 ;self.A[5, 11] = -1.0
            self.b = np.zeros(35)
            self.solve_grf_eef = self.solve_SRB_grf_eef_simple
        elif solver == "DYNAMICS":
            self.A = np.zeros((38, 12))
            self.A[0:3, 6:9] = np.eye(3); self.A[0:3, 9:12] = np.eye(3)
            self.A[2, 8] = 1.0 ;self.A[2, 11] = -1.0
            self.b = np.zeros(38)
            self.solve_grf_eef = self.solve_DYN_grf_eef
        elif solver == "DYNAMICS_Z":
            self.A = np.zeros((36, 4))
            self.A[0, 2] = 1; self.A[0, 3] = -1
            self.b = np.zeros(36)
            self.solve_grf_eef = self.solve_DYN_Z_grf_eef
        else:
            raise ValueError("Invalid SRB solver type")

    def run(self):
        while True:
            robot_state_data = self.robot_state_data_shm.copy()
            if robot_state_data is not None:
                # Get whole body qpos, qvel, qtau
                qpos = robot_state_data[:, :7+self.num_dofs]
                qvel = robot_state_data[:, 7+self.num_dofs:7+self.num_dofs+6+self.num_dofs]
                qtau = robot_state_data[:, 7+self.num_dofs+6+self.num_dofs:7+self.num_dofs+6+self.num_dofs+self.num_dofs+6]
                qacc = robot_state_data[:, 7+self.num_dofs+6+self.num_dofs+self.num_dofs+6:]
                GRF, EEF, info = self.UpdateEstimation(qpos[0], qvel[0], qacc[0], qtau[0])
                self.EEF_z = EEF[2]
                self.GRF_z = GRF[0][2] + GRF[1][2] - 9.81 * self.robot_mass
                print("EEF: ", EEF)
                self.estimated_ee_force_shm[0] = EEF
            
            self.rate.sleep()

    def UpdateEstimation(self, qpos, qvel, qacc, tau):
        # Quaternion: wxyz to xyzw
        qpos[3:7] = quat_wxyz_to_xyzw(qpos[3:7])
        # Get the GRF (No forces on the hands)
        # GRF = self.solve_SRB_grf(qpos, qvel)
        # print("GRF_left: ", GRF[0], " GRF_right: ", GRF)
        # Get the GRF and EEF
        GRF, EEF, info = self.solve_grf_eef(qpos, qvel, qacc, tau)
        
        # Visualize the robot in meshcat
        if self.vis_pin: self.vis.display(qpos)
        
        return GRF, EEF, info
    
    def solve_SRB_grf(self, qpos, qvel):
        """
        Solve the SRB for ground reaction force (grf).
        Consider one force for each foot:
        [1_3        1_3      ]       [m * com_acc + g]
        [(p_1-p_c)x (p_2-p_c)x] * F = [Ig_w * w_b_w]
        
        Returns: F_left, F_right
        """
        # a = pin.utils.zero(self.robot.model.nv)
        # v = pin.utils.zero(self.robot.model.nv)
        A = np.zeros((6, 6))
        A[0:3, 0:3] = np.eye(3)
        A[0:3, 3:6] = np.eye(3)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        p_1 = self.robot.data.oMf[self.frames_idx['left_ankle_roll_link']].translation
        p_2 = self.robot.data.oMf[self.frames_idx['right_ankle_roll_link']].translation
        # Get the CoM info
        com_pos, com_vel, com_acc = self.calculate_com(qpos, qvel)
        # Get the centroidal momentum
        Ig_w = self.caculate_moi(qpos, qvel)
        p_1c = p_1 - com_pos
        p_2c = p_2 - com_pos
        A[3:6, 0:3] = np.array([[0, -p_1c[2], p_1c[1]],
                                [p_1c[2], 0, -p_1c[0]],
                                [-p_1c[1], p_1c[0], 0]])
        A[3:6, 3:6] = np.array([[0, -p_2c[2], p_2c[1]],
                                [p_2c[2], 0, -p_2c[0]],
                                [-p_2c[1], p_2c[0], 0]])
        b = np.zeros(6)
        b[0:3] = self.robot_mass * (com_acc - np.array([0, 0, -9.81]))
        # Get the angular velocity in the world frame (qvel is in the base frame)
        w_b_c = qvel[3:6]
        pin.forwardKinematics(self.robot.model, self.robot.data, qpos, qvel)
        Rw_c = self.robot.data.oMi[1].rotation
        w_b_w = Rw_c @ w_b_c
        b[3:6] = Ig_w @ w_b_w
        lambda_reg = 1e-4
        F = np.linalg.inv(A.T @ A + lambda_reg * np.eye(A.shape[1])) @ (A.T @ b)
        # print("F: ", F)
        return F[0:3], F[3:6]
    
    def solve_SRB_grf_eef_full(self, qpos, qvel, qacc, tau):
        """
        Solve the SRB for ground reaction force (grf) and end-effector force (eef).
        Consider one force for each foot:

        F = [F_1, F_2, F_3, F_4], GRF = [F_1, F_2], EEF = F_3 + F_4

        [1_3        1_3      1_3      1_3      ]            [m * com_acc + g]
        [(p_1-p_c)x (p_2-p_c)x (p_3-p_c)x (p_4-p_c)x] * F = [Ig_w * w_b_w]
        [0_3        0_3      1_3      diag(1,1,-1)      ]            [0, 0, 0]
        [                J^T * J               ]            [J^T * H(q,qdot,qddot,tau)]
         
        Returns: GRF, EEF
        """
        GRF = np.zeros((2, 3))
        EEF = np.zeros(3)
        # Get the CoM info and update the model kinematics/frame placements
        pin.forwardKinematics(self.robot.model, self.robot.data, qpos, qvel)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        com_pos, com_vel, com_acc = self.calculate_com(qpos, qvel, qacc)
        p_1 = self.robot.data.oMf[self.frames_idx['left_ankle_roll_link']].translation
        p_2 = self.robot.data.oMf[self.frames_idx['right_ankle_roll_link']].translation
        p_3 = self.robot.data.oMf[self.frames_idx['left_rubber_hand']].translation
        p_4 = self.robot.data.oMf[self.frames_idx['right_rubber_hand']].translation
        # Get the centroidal momentum
        # Ig_w = self.caculate_moi(qpos, qvel)
        # Get the Jacobians (linear part only)
        J_1 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['left_ankle_roll_link'])[0:3, 6:]
        J_2 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['right_ankle_roll_link'])[0:3, 6:]
        J_3 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['left_rubber_hand'])[0:3, 6:]
        J_4 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['right_rubber_hand'])[0:3, 6:]
        J_T = np.concatenate((J_1.T, J_2.T, J_3.T, J_4.T), axis=1)
        # print("J_1: ", J_1)

        # Construct the A matrix
        p_1c = p_1 - com_pos
        p_2c = p_2 - com_pos
        p_3c = p_3 - com_pos
        p_4c = p_4 - com_pos
        self.A[3:6, 0:3] = skew_symmetric(p_1c)
        self.A[3:6, 3:6] = skew_symmetric(p_2c)
        self.A[3:6, 6:9] = skew_symmetric(p_3c)
        self.A[3:6, 9:12] = skew_symmetric(p_4c)
        self.A[9:, :] = J_T.T @ J_T
        # Construct the b vector
        self.b[0:3] = self.robot_mass * (com_acc - np.array([0, 0, -9.81]))
        # Get the angular acceleration in the world frame (qacc is in the base frame)
        # wdot_b_c = np.array([0.0, 0.0, qacc[5]])
        # # pin.forwardKinematics(self.robot.model, self.robot.data, qpos, qvel)
        # Rw_c = self.robot.data.oMi[1].rotation
        # w_b_w = Rw_c @ wdot_b_c
        # self.b[3:6] = Ig_w @ w_b_w
        self.b[3:6] = np.zeros(3)
        # Compute the dynamics
        compensation = self.caculate_compensation(qpos, qvel, qacc)
        self.b[9:] = J_T.T @ (compensation - tau)[6:]

        # Solve the SRB + Dynamics
        lambda_reg = 1e-6
        F = np.linalg.inv(self.A.T @ self.A + lambda_reg * np.eye(self.A.shape[1])) @ (self.A.T @ self.b)
        GRF[0] = F[0:3]
        GRF[1] = F[3:6]
        EEF = F[6:9] + F[9:12]
        if EEF[0] > 100:
            print("qpos: ", qpos)
            print("qvel: ", qvel)
            print("qacc: ", qacc)
        # print("F[6:9]: ", F[6:9], " F[9:12]: ", F[9:12])
        print("EEF: ", EEF)
        # print("GRF: ", GRF, " EEF: ", EEF)
        # print("com_pos: ", com_pos, " p_1: ", p_1, " p_2: ", p_2, " p_3: ", p_3, " p_4: ", p_4)
        return GRF, EEF, [com_pos, p_1, p_2, p_3, p_4]

    def solve_SRB_grf_eef_simple(self, qpos, qvel, qacc, tau):
        """
        Solve the SRB for ground reaction force (grf) and end-effector force (eef).
        Consider one force for each foot:

        F = [F_1, F_2, F_3, F_4], GRF = [F_1, F_2], EEF = F_3 + F_4

        [1_3        1_3      1_3      1_3      ]            [m * com_acc + g]
        [0_3        0_3      1_3      diag(1,1,-1)      ]            [0, 0, 0]
        [                J               ]            [H(q,qdot,qddot,tau)]
         
        Returns: GRF, EEF
        """
        GRF = np.zeros((2, 3))
        EEF = np.zeros(3)
        # Get the CoM info and update the model kinematics/frame placements
        pin.forwardKinematics(self.robot.model, self.robot.data, qpos, qvel)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        com_pos, com_vel, com_acc = self.calculate_com(qpos, qvel, qacc)
        # Get the Jacobians (linear part only)
        J_1 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['left_ankle_roll_link'])[0:3, 6:]
        J_2 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['right_ankle_roll_link'])[0:3, 6:]
        J_3 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['left_rubber_hand'])[0:3, 6:]
        J_4 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['right_rubber_hand'])[0:3, 6:]
        J_T = np.concatenate((J_1.T, J_2.T, J_3.T, J_4.T), axis=1)
        # print("J_1: ", J_1)

        # Construct the A matrix
        self.A[6:, :] = J_T
        # Construct the b vector
        # com_acc[0:2] *= 0.0
        self.b[0:3] = self.robot_mass * (com_acc - np.array([0, 0, -9.81]))
        # Compute the dynamics
        compensation = self.caculate_compensation(qpos, qvel, qacc)
        self.b[6:] = (compensation - tau)[6:]

        # Solve the SRB + Dynamics
        lambda_reg = 1e-6
        F = np.linalg.inv(self.A.T @ self.A) @ (self.A.T @ self.b)
        GRF[0] = F[0:3]
        GRF[1] = F[3:6]
        EEF = F[6:9] + F[9:12]
        # if EEF[0] > 100:
        #     print("qpos: ", qpos)
        #     print("qvel: ", qvel)
        #     print("qacc: ", qacc)
        # print("F[6:9]: ", F[6:9], " F[9:12]: ", F[9:12])
        # print("EEF: ", EEF)
        # print("GRF: ", GRF, " EEF: ", EEF)
        return GRF, EEF, None

    def solve_DYN_grf_eef(self, qpos, qvel, qacc, tau):
        """
        Solve the SRB for ground reaction force (grf) and end-effector force (eef).
        Consider one force for each foot:

        F = [F_1, F_2, F_3, F_4], GRF = [F_1, F_2], EEF = F_3 + F_4

        [0_3        0_3      1_3      diag(1,1,-1)      ]            [0, 0, 0]
        [                J^T * J               ]            [J^T * H(q,qdot,qddot,tau)]
         
        Returns: GRF, EEF
        """
        GRF = np.zeros((2, 3))
        EEF = np.zeros(3)
        # Get the CoM info and update the model kinematics/frame placements
        pin.forwardKinematics(self.robot.model, self.robot.data, qpos, qvel)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        # Get the Jacobians (linear part only)
        J_1 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['left_ankle_roll_link'])[0:3, :]
        J_2 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['right_ankle_roll_link'])[0:3, :]
        J_3 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['left_rubber_hand'])[0:3, :]
        J_4 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['right_rubber_hand'])[0:3, :]
        J_T = np.concatenate((J_1.T, J_2.T, J_3.T, J_4.T), axis=1)
        # print("J_1: ", J_1)

        # Construct the A matrix
        # print("self.A[3:, :].shape: ", self.A[3:, :].shape, " J_T.T.shape: ", J_T.T.shape)
        self.A[3:, :] = J_T # Shape: (35, 12)
        # Construct the b vector
        # Compute the dynamics
        compensation = self.caculate_compensation(qpos, qvel, qacc)
        self.b[3:] = (compensation - tau) # Shape: (35,)

        # Solve the SRB + Dynamics
        lambda_reg = 1e-6
        F = np.linalg.inv(self.A.T @ self.A + lambda_reg * np.eye(self.A.shape[1])) @ (self.A.T @ self.b)
        GRF[0] = F[0:3]
        GRF[1] = F[3:6]
        EEF = F[6:9] + F[9:12]
        if EEF[0] > 100:
            print("qpos: ", qpos)
            print("qvel: ", qvel)
            print("qacc: ", qacc)
        # print("F[6:9]: ", F[6:9], " F[9:12]: ", F[9:12])
        print("EEF: ", EEF)
        # print("GRF: ", GRF, " EEF: ", EEF)
        return GRF, EEF, None

    def solve_DYN_Z_grf_eef(self, qpos, qvel, qacc, tau):
        """
        Solve the SRB for ground reaction force (grf) and end-effector force (eef).
        Consider one force for each foot:

        F = [F_1, F_2, F_3, F_4], GRF = [F_1, F_2], EEF = F_3 + F_4

        [0        0      1      -1      ]            [0]
        [                J^T * J               ]     [J^T * H(q,qdot,qddot,tau)]
         
        Returns: GRF, EEF
        """
        GRF = np.zeros((2, 3))
        EEF = np.zeros(3)
        # Get the CoM info and update the model kinematics/frame placements
        pin.forwardKinematics(self.robot.model, self.robot.data, qpos, qvel)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        # Get the Jacobians (linear part only)
        J_1 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['left_ankle_roll_link'])[2:3, :]
        J_2 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['right_ankle_roll_link'])[2:3, :]
        J_3 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['left_rubber_hand'])[2:3, :]
        J_4 = pin.computeFrameJacobian(self.robot.model, self.robot.data, qpos, self.frames_idx['right_rubber_hand'])[2:3, :]
        J_T = np.concatenate((J_1.T, J_2.T, J_3.T, J_4.T), axis=1)
        # print("J_1: ", J_1)

        # Construct the A matrix
        # print("self.A[3:, :].shape: ", self.A[3:, :].shape, " J_T.T.shape: ", J_T.T.shape)
        self.A[1:, :] = J_T # Shape: (35, 12)
        # Construct the b vector
        # Compute the dynamics
        compensation = self.caculate_compensation(qpos, qvel, qacc)
        self.b[1:] = (compensation - tau) # Shape: (35,)

        # Solve the SRB + Dynamics
        lambda_reg = 1e-6
        F = np.linalg.inv(self.A.T @ self.A + lambda_reg * np.eye(self.A.shape[1])) @ (self.A.T @ self.b)
        GRF[0, 2] = F[0]
        GRF[1, 2] = F[1]
        EEF[2] = F[2] + F[3]
        if EEF[0] > 100:
            print("qpos: ", qpos)
            print("qvel: ", qvel)
            print("qacc: ", qacc)
        # print("F[6:9]: ", F[6:9], " F[9:12]: ", F[9:12])
        print("EEF: ", EEF)
        # print("GRF: ", GRF, " EEF: ", EEF)
        return GRF, EEF, None

    def calculate_com(self, qpos, qvel, qacc=None):
        pin.centerOfMass(self.robot.model, self.robot.data, qpos, qvel, qacc)
        # print("qvel[2]: ", self.mj_data.qvel[2], " v_CoM: ", self.robot.data.vcom[0][2])
        return self.robot.data.com[0], self.robot.data.vcom[0], self.robot.data.acom[0]
    
    def caculate_moi(self, q, v):
        # update centroidal momentum
        h = pin.ccrba(self.robot.model, self.robot.data, q, v)
        # print("Centroidal Momentum: ", h)
        Ig_w = self.robot.data.Ig.inertia
        return Ig_w

    def caculate_compensation(self, qpos, qvel, qacc=None):
        if qacc is None:
            qacc = np.zeros(self.num_dofs+6)
        # Gq = pin.computeGeneralizedGravity(self.robot.model, self.robot.data, qpos)
        # print("Generalized Gravity: ", Gq)
        compensation = pin.rnea(self.robot.model, self.robot.data, qpos, qvel, qacc)
        # print("Compensation: ", compensation)
        return compensation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/g1_29dof_free_force.yaml', help='config file')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    force_estimator = ForceEstimator(config)
    force_estimator.run()