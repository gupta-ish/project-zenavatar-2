import rclpy
from rclpy.node import Node
import numpy as np
import time
from std_msgs.msg import Float64MultiArray
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize

from unitree_sdk2py.utils.crc import CRC
import threading
import argparse
import yaml
# import sys
# sys.path.append('../../')
from pynput import keyboard
from sim2real.utils.robot import Robot
from termcolor import colored

class CommandSender:
    def __init__(self, config, node):
        self.config = config 
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        elif self.config["ROBOT_TYPE"] == "g1_29dof" or self.config["ROBOT_TYPE"] == "h1-2_21dof" or self.config["ROBOT_TYPE"] == "h1-2_27dof":
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        else:
            raise NotImplementedError(f"Robot type {self.config['ROBOT_TYPE']} is not supported yet")

        self.robot = Robot(config)
        self.kp_level = 0.1
        self.robot_kp = np.zeros(self.robot.NUM_MOTORS)
        self.robot_kd = np.zeros(self.robot.NUM_MOTORS)
        for i in range(len(self.robot.MOTOR_KP)):
            self.robot_kp[i] = self.robot.MOTOR_KP[i] * self.kp_level

        for i in range(len(self.robot.MOTOR_KD)):
            self.robot_kd[i] = self.robot.MOTOR_KD[i] * 1.0

        self.weak_motor_joint_index = []
        for key, value in self.robot.WeakMotorJointIndex.items():
            self.weak_motor_joint_index.append(value)

        self.node = node

        # init low cmd publisher
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # subscribe to the robot command by rl inference
        self.commands_sub = self.node.create_subscription(
            Float64MultiArray, "robot_command", self.command_callback, 1
        )
        self.command_msg = None

        self.send_command_q = np.zeros(self.robot.NUM_JOINTS)
        self.send_command_dq = np.zeros(self.robot.NUM_JOINTS)
        self.send_command_tau = np.zeros(self.robot.NUM_JOINTS)

        if self.config["ROBOT_TYPE"]=="h1" or self.config["ROBOT_TYPE"]=="go2":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        elif self.config["ROBOT_TYPE"] == "g1_29dof" or self.config["ROBOT_TYPE"] == "h1-2_21dof" or self.config["ROBOT_TYPE"] == "h1-2_27dof":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        else:
            raise NotImplementedError(f"Robot type {self.config['ROBOT_TYPE']} is not supported yet")

        self.InitLowCmd()
        self.low_state = None
        self.crc = CRC()

        self.timestamp_digit = 6
        self.timestamp_message = [0.0] * self.timestamp_digit

        self.get_command_timestep = 0.0
        self.send_command_timestep = 0.0

        self.delay_timestamp_publisher = self.node.create_publisher(Float64MultiArray, "delay_timestamp", 1)

        self.command_joint_pos = Float64MultiArray(data=[0.0]*self.robot.NUM_JOINTS)
        self.command_joint_pos_pub = self.node.create_publisher(Float64MultiArray, "command_joint_pos", 1)
        # Yuanhang: command_joint_vel and command_joint_torque is not used in the current version
        self.command_joint_vel = Float64MultiArray(data=[0.0]*self.robot.NUM_JOINTS)
        self.command_joint_vel_pub = self.node.create_publisher(Float64MultiArray, "command_joint_vel", 1)
        self.command_joint_torque = Float64MultiArray(data=[0.0]*self.robot.NUM_JOINTS)
        self.command_joint_torque_pub = self.node.create_publisher(Float64MultiArray, "command_joint_torque", 1)

        self.debug_joint_index = 0
        self.debug_amplitude = 0.2
        
        self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
        self.key_listener_thread.start()




    def start_key_listener(self):
        """Start a key listener using pynput."""
        def on_press(key):
            try:
                if key.char == "1":
                    self.debug_joint_index += 1
                    self.kp_level = 0.1
                    for i in range(len(self.robot.MOTOR_KP)):
                        self.robot_55555555kp[i] = self.robot.MOTOR_KP[i] * self.kp_level
                    self.debug_joint_index %= self.robot.NUM_MOTORS
                    self.node.get_logger().info(colored(f"Debug joint index: {self.debug_joint_index}", "green"))
                elif key.char == "2":
                    self.debug_joint_index -= 1
                    self.kp_level = 0.1
                    for i in range(len(self.robot.MOTOR_KP)):
                        self.robot_kp[i] = self.robot.MOTOR_KP[i] * self.kp_level
                    self.debug_joint_index %= self.robot.NUM_MOTORS
                    self.node.get_logger().info(colored(f"Debug joint index: {self.debug_joint_index}", "green"))
                elif key.char == "3":
                    self.debug_amplitude += 0.1
                    self.node.get_logger().info(colored(f"Debug amplitude: {self.debug_amplitude}", "green"))
                elif key.char == "4":
                    self.debug_amplitude -= 0.1
                    self.node.get_logger().info(colored(f"Debug amplitude: {self.debug_amplitude}", "green"))
                elif key.char == "5":
                    self.kp_level += 0.1
                    for i in range(len(self.robot.MOTOR_KP)):
                        self.robot_kp[i] = self.robot.MOTOR_KP[i] * self.kp_level
                    self.node.get_logger().info(colored(f"Debug kp level: {self.kp_level}", "green"))
                elif key.char == "6":
                    self.kp_level -= 0.1
                    for i in range(len(self.robot.MOTOR_KP)):
                        self.robot_kp[i] = self.robot.MOTOR_KP[i] * self.kp_level
                    self.node.get_logger().info(colored(f"Debug kp level: {self.kp_level}", "green"))

            except AttributeError:
                pass  # Handle special keys if needed

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        listener.join()  # Keep the thread alive


    def InitLowCmd(self):
        # h1/go2:
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            self.low_cmd.head[0] = 0xFE
            self.low_cmd.head[1] = 0xEF
        else:
            pass

        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(self.robot.NUM_MOTORS):
            if self.is_weak_motor(i):
                self.low_cmd.motor_cmd[i].mode = 0x01 
            else:
                self.low_cmd.motor_cmd[i].mode = 0x0A 
            self.low_cmd.motor_cmd[i].q= self.robot.UNITREE_LEGGED_CONST["PosStopF"]
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = self.robot.UNITREE_LEGGED_CONST["VelStopF"]
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0
            if self.config["ROBOT_TYPE"] == "g1_29dof" or self.config["ROBOT_TYPE"] == "h1-2_21dof" or self.config["ROBOT_TYPE"] == "h1-2_27dof":
                self.low_cmd.mode_machine = self.config["UNITREE_LEGGED_CONST"]["MODE_MACHINE"]
                self.low_cmd.mode_pr = self.config["UNITREE_LEGGED_CONST"]["MODE_PR"]
            else:
                pass

    def is_weak_motor(self, motor_index):
        return motor_index in self.weak_motor_joint_index

    def command_callback(self, msg):
        self.command_msg = msg
        self.get_command_timestep = self.node.get_clock().now().nanoseconds / 1e9


    def _prepare_low_cmd(self):
        self.timestamp_message[0] = self.command_msg.data[0]
        self.timestamp_message[1] = self.command_msg.data[1]
        self.timestamp_message[2] = self.command_msg.data[2]
        self.timestamp_message[3] = self.command_msg.data[3]
        self.timestamp_message[4] = self.get_command_timestep

        self.emergence_stop = self.command_msg.data[self.timestamp_digit]
        # print("emergence_stop: ", self.emergence_stop)

        # Initialize command arrays with zeros
        self.send_command_q = np.zeros(self.robot.NUM_MOTORS)

        # Generate sine wave for a specific joint index
        # You can adjust these parameters as needed
        amplitude = self.debug_amplitude  # radians
        frequency = 0.5  # Hz
        joint_index = self.debug_joint_index  # The joint you want to control

        # Calculate sine wave based on time
        current_time = time.time()
        self.send_command_q[joint_index] = amplitude * np.sin(2 * np.pi * frequency * current_time)
        self.send_command_dq = np.zeros(self.robot.NUM_MOTORS)
        self.send_command_tau = np.zeros(self.robot.NUM_MOTORS)

    def send_command(self):
        for i in range(self.robot.NUM_MOTORS):
            motor_index = self.robot.JOINT2MOTOR[i]
            self.low_cmd.motor_cmd[motor_index].q = self.send_command_q[motor_index]
            self.low_cmd.motor_cmd[motor_index].dq = self.send_command_dq[motor_index]
            self.low_cmd.motor_cmd[motor_index].tau = self.send_command_tau[motor_index]
            # kp kd
            if self.emergence_stop:
                self.low_cmd.motor_cmd[motor_index].kp = 0.0
            else:
                self.low_cmd.motor_cmd[motor_index].kp = self.robot_kp[motor_index]
            self.low_cmd.motor_cmd[motor_index].kd = self.robot_kd[motor_index]


        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

        # publish command joint
        self.command_joint_pos.data = self.send_command_q.tolist()
        self.command_joint_pos_pub.publish(self.command_joint_pos)
        # TODO: publish command joint vel and torque
        self.command_joint_vel.data = self.send_command_dq.tolist()
        self.command_joint_vel_pub.publish(self.command_joint_vel)
        self.command_joint_torque.data = self.send_command_tau.tolist()
        self.command_joint_torque_pub.publish(self.command_joint_torque)

    def main_loop(self):
        self._prepare_low_cmd()
        self.send_command_timestep = self.node.get_clock().now().nanoseconds / 1e9
        self.timestamp_message[5] = self.send_command_timestep
        self.delay_timestamp_publisher.publish(Float64MultiArray(data=self.timestamp_message))

        self.send_command()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    parser.add_argument('--onnx_model_path', type=str, default=None, help='onnx model path')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    rclpy.init(args=None)
    node = rclpy.create_node('command_sender')

    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()

    rate = node.create_rate(200)

    ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    command_sender = CommandSender(config, node)
    time.sleep(1)

    start_time = time.time()
    command_cnt = 0

    try:
        while rclpy.ok():
            command_sender.main_loop()
            rate.sleep()
            command_cnt += 1
            if command_cnt % 100 == 0:
                node.get_logger().info(f"Average command_cnt FPS: {100/(time.time()-start_time)}")
                start_time = time.time()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()