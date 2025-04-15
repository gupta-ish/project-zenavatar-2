
import numpy as np
import time
from std_msgs.msg import Float64MultiArray

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

import argparse
import yaml
from loguru import logger
from utils.robot import Robot
import threading


import rclpy
from rclpy.node import Node

class StateCommandPublisher(Node):
    def __init__(self, config):
        super().__init__("StatePublisher")
        # subscribe to the robot state by unitree sdk
        self.config = config
        self.robot = Robot(config)
        self.robot_low_state = None

        self.num_dof = self.robot.NUM_JOINTS
        # 3 + 4 + 19
        self._init_q = np.zeros(3 + 4 + self.num_dof)
        self.q = self._init_q
        self.dq = np.zeros(3 + 3 + self.num_dof)
        self.tau_est = np.zeros(self.num_dof)
        self.temp_first = np.zeros(self.num_dof)
        self.temp_second = np.zeros(self.num_dof)

        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_go, 1)
        elif self.config["ROBOT_TYPE"] == "g1_29dof" or self.config["ROBOT_TYPE"] == "h1-2_27dof" or self.config["ROBOT_TYPE"] == "h1-2_21dof":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_hg)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_hg, 1)
        else:
            raise NotImplementedError(f"Robot type {self.config['ROBOT_TYPE']} is not supported")
        
        # Initialize ROS state publishers for foxglove visualization
        self.send_joint_pos = Float64MultiArray(data=[0.0]*self.num_dof)
        self.send_joint_pos_pub = self.create_publisher(Float64MultiArray, "robot_joint_pos", 1)
        self.send_joint_vel = Float64MultiArray(data=[0.0]*self.num_dof)
        self.send_joint_vel_pub = self.create_publisher(Float64MultiArray, "robot_joint_vel", 1)
        self.send_joint_torque = Float64MultiArray(data=[0.0]*self.num_dof)
        self.send_joint_torque_pub = self.create_publisher(Float64MultiArray, "robot_joint_torque", 1)
        self.send_joint_temp_first = Float64MultiArray(data=[0.0]*self.num_dof)
        self.send_joint_temp_first_pub = self.create_publisher(Float64MultiArray, "robot_joint_temp_first", 1)
        self.send_joint_temp_second = Float64MultiArray(data=[0.0]*self.num_dof)
        self.send_joint_temp_second_pub = self.create_publisher(Float64MultiArray, "robot_joint_temp_second", 1)
        
    def LowStateHandler_go(self, msg: LowState_go): # Yuanhang: the msg type MUST be declared explicitly for simulation
        self.robot_low_state = msg
    
    def LowStateHandler_hg(self, msg: LowState_hg): # Yuanhang: the msg type MUST be declared explicitly for simulation
        self.robot_low_state = msg

    def _prepare_low_state(self):
        imu_state = self.robot_low_state.imu_state
        # base quaternion
        self.q[0:3] = 0.0
        self.q[3:7] = imu_state.quaternion # w, x, y, z
        self.dq[3:6] = imu_state.gyroscope
        unitree_joint_state = self.robot_low_state.motor_state

        for i in range(self.num_dof):
            # import ipdb; ipdb.set_trace()
            self.q[7+i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].q
            self.dq[6+i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].dq
            self.tau_est[i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].tau_est
            self.temp_first[i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].temperature[0]
            self.temp_second[i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].temperature[1]

    def main_loop(self,):
        self._prepare_low_state()
        # print(f"Send state to ROS2: {self.publish_to_ros_timestep - self.receive_from_sdk_timestep}")
        # print(f"Time in message: {state_msg.data[1] - state_msg.data[0]}")
        self.send_joint_pos.data = self.q[7:].tolist()
        self.send_joint_pos_pub.publish(self.send_joint_pos)
        self.send_joint_vel.data = self.dq[6:].tolist()
        self.send_joint_vel_pub.publish(self.send_joint_vel)
        self.send_joint_torque.data = self.tau_est.tolist()
        self.send_joint_torque_pub.publish(self.send_joint_torque)
        self.send_joint_temp_first.data = self.temp_first.tolist()
        self.send_joint_temp_first_pub.publish(self.send_joint_temp_first)
        self.send_joint_temp_second.data = self.temp_second.tolist()
        self.send_joint_temp_second_pub.publish(self.send_joint_temp_second)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    rclpy.init(args=None)

    if not config.get("INTERFACE", None):
        ChannelFactoryInitialize(config["DOMAIN_ID"])
    else: ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])

    state_publisher = StatePublisher(config)

    thread = threading.Thread(target=rclpy.spin, args=(state_publisher, ), daemon=True)
    thread.start()

    freq = 1000
    rate = state_publisher.create_rate(freq)
    if freq != 1000:
        # warning before start
        state_publisher.get_logger().warning(f"State publisher is running at {freq}Hz, not 120Hz")
        # ask for confirmation
        state_publisher.get_logger().warning("Please make sure the frequency is consistent with the simulation")
        import ipdb; ipdb.set_trace()
    start_time = time.time()
    total_publish_cnt = 0
    try:
        while rclpy.ok():
            state_publisher.main_loop()
            total_publish_cnt += 1
            if total_publish_cnt % 500 == 0:
                end_time = time.time()
                # self.get_logger().info(f"state sent {state_msg.data}")
                state_publisher.get_logger().info(f"FPS: {500/(end_time - start_time)}")
                start_time = end_time
            rate.sleep()
    except KeyboardInterrupt:
        pass
    state_publisher.destroy_node()
    rclpy.shutdown()