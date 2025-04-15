
import numpy as np
import time
from std_msgs.msg import Float64MultiArray

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmd_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmd_hg

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import argparse
import yaml
from loguru import logger
from utils.robot import Robot
import threading


import rclpy
from rclpy.node import Node

class CommandPublisher(Node):
    def __init__(self, config):
        super().__init__("CommandPublisher")
        # subscribe to the robot state by unitree sdk
        self.config = config
        self.robot = Robot(config)
        self.robot_low_cmd = None
        
        self.num_dof = self.robot.NUM_JOINTS

        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            self.robot_command_subscriber = ChannelSubscriber("rt/command", LowCmd_go)
            self.robot_command_subscriber.Init(self.LowCmdHandler_go, 1)
        elif self.config["ROBOT_TYPE"] == "g1_29dof" or self.config["ROBOT_TYPE"] == "h1-2_27dof" or self.config["ROBOT_TYPE"] == "h1-2_21dof":
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            self.robot_command_subscriber = ChannelSubscriber("rt/command", LowCmd_hg)
            self.robot_command_subscriber.Init(self.LowCmdHandler_hg, 1)
        else:
            raise NotImplementedError(f"Robot type {self.config['ROBOT_TYPE']} is not supported")

        self.send_command_q = np.zeros(self.robot.NUM_JOINTS)
        self.send_command_dq = np.zeros(self.robot.NUM_JOINTS)
        self.send_command_tau = np.zeros(self.robot.NUM_JOINTS)
        
        # Initialize ROS state publishers for foxglove visualization
        self.command_joint_pos = Float64MultiArray(data=[0.0]*self.robot.NUM_JOINTS)
        self.command_joint_pos_pub = self.node.create_publisher(Float64MultiArray, "command_joint_pos", 1)
        # Yuanhang: command_joint_vel and command_joint_torque is not used in the current version
        self.command_joint_vel = Float64MultiArray(data=[0.0]*self.robot.NUM_JOINTS)
        self.command_joint_vel_pub = self.node.create_publisher(Float64MultiArray, "command_joint_vel", 1)
        self.command_joint_torque = Float64MultiArray(data=[0.0]*self.robot.NUM_JOINTS)
        self.command_joint_torque_pub = self.node.create_publisher(Float64MultiArray, "command_joint_torque", 1)
        
    def LowCmdHandler_go(self, msg: LowCmd_go):
        self.robot_low_cmd = msg
    
    def LowCmdHandler_hg(self, msg: LowCmd_hg):
        self.robot_low_cmd = msg

    def _prepare_command(self,):
        # receive command from ROS2
        self.send_command_q = self.robot_low_cmd.q
        self.send_command_dq = self.robot_low_cmd.dq
        self.send_command_tau = self.robot_low_cmd.tau
    
    def main_loop(self,):
        self._prepare_command()
        # publish command joint
        self.command_joint_pos.data = self.send_command_q.tolist()
        self.command_joint_pos_pub.publish(self.command_joint_pos)
        # TODO: publish command joint vel and torque
        self.command_joint_vel.data = self.send_command_dq.tolist()
        self.command_joint_vel_pub.publish(self.command_joint_vel)
        self.command_joint_torque.data = self.send_command_tau.tolist()
        self.command_joint_torque_pub.publish(self.command_joint_torque)

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

    publisher = CommandPublisher(config)

    thread = threading.Thread(target=rclpy.spin, args=(publisher, ), daemon=True)
    thread.start()

    freq = 100
    rate = publisher.create_rate(freq)
    start_time = time.time()
    total_publish_cnt = 0
    try:
        while rclpy.ok():
            publisher.main_loop()
            total_publish_cnt += 1
            if total_publish_cnt % 500 == 0:
                end_time = time.time()
                # self.get_logger().info(f"state sent {state_msg.data}")
                publisher.get_logger().info(f"FPS: {500/(end_time - start_time)}")
                start_time = end_time
            rate.sleep()
    except KeyboardInterrupt:
        pass
    publisher.destroy_node()
    rclpy.shutdown()