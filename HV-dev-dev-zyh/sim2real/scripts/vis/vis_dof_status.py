import rclpy
import threading
from std_msgs.msg import Float64MultiArray, Float32
import yaml
import argparse
import time
import numpy as np
from collections import deque

class VisJoint:
    def __init__(self, node, config):
        self.node = node
        self.num_joints = len(config['JOINT_NAME_LIST'])
        self.joint_name_list = config['JOINT_NAME_LIST']

        self.max_joint_temp = 48.0

        

        # Publishers for joint status arrays
        self.joint_state_pubs = {
            attr: {
                'current': {
                    joint_name: self.node.create_publisher(Float32, f"/dof/joint_{joint_name}_{attr}_current", 1)
                    for joint_name in self.joint_name_list
                },
                'command': {
                    joint_name: self.node.create_publisher(Float32, f"/dof/joint_{joint_name}_{attr}_command", 1)
                    for joint_name in self.joint_name_list
                }
            }
            for attr in ['pos', 'vel', 'torque']
        }
        # Placeholder for received joint data
        self.joint_data_msgs = {
            'pos': None,
            'vel': None,
            'torque': None,
            'temp': None,
            'cmd_pos': None,
            'cmd_vel': None,
            'cmd_torque': None, # Yuanhang: cmd_torque is the same as torque
        }

        # Joint limits
        self.joint_limits = {
            'pos': (config['dof_pos_lower_limit_list'], config['dof_pos_upper_limit_list']),
            'vel': ([-lim for lim in config['dof_vel_limit_list']], config['dof_vel_limit_list']),
            'torque': ([-lim for lim in config['dof_effort_limit_list']], config['dof_effort_limit_list']),
            'temp': ([-100.0]*self.num_joints, [self.max_joint_temp] * self.num_joints),
        }

        # Publishers for joint limits (lower and upper)
        self.joint_limit_pubs = {
            attr: {
                'lower': {
                    joint_name: self.node.create_publisher(Float32, f"/dof/joint_{joint_name}_{attr}_lower_limit", 1)
                    for joint_name in self.joint_name_list
                },
                'upper': {
                    joint_name: self.node.create_publisher(Float32, f"/dof/joint_{joint_name}_{attr}_upper_limit", 1)
                    for joint_name in self.joint_name_list
                }
            }
            for attr in self.joint_limits
        }

        # Initialize subscribers
        self.robot_joint_pos_sub = self.node.create_subscription(
            Float64MultiArray, "robot_joint_pos", self.robot_joint_pos_callback, 1
        )
        self.robot_joint_vel_sub = self.node.create_subscription(
            Float64MultiArray, "robot_joint_vel", self.robot_joint_vel_callback, 1
        )
        self.robot_joint_torque_sub = self.node.create_subscription(
            Float64MultiArray, "robot_joint_torque", self.robot_joint_torque_callback, 1
        )
        self.robot_joint_temp_first_sub = self.node.create_subscription(
            Float64MultiArray, "robot_joint_temp_first", self.robot_joint_temp_callback, 1
        )
        self.command_joint_pos_sub = self.node.create_subscription(
            Float64MultiArray, "command_joint_pos", self.command_joint_pos_callback, 1
        )
        self.command_joint_vel_sub = self.node.create_subscription(
            Float64MultiArray, "command_joint_vel", self.command_joint_vel_callback, 1
        )

    # Callbacks for subscriptions
    def robot_joint_pos_callback(self, msg):
        self.joint_data_msgs['pos'] = msg

    def robot_joint_vel_callback(self, msg):
        self.joint_data_msgs['vel'] = msg

    def robot_joint_torque_callback(self, msg):
        self.joint_data_msgs['torque'] = msg
        self.joint_data_msgs['cmd_torque'] = msg

    def robot_joint_temp_callback(self, msg):
        self.joint_data_msgs['temp'] = msg
    
    def command_joint_pos_callback(self, msg):
        self.joint_data_msgs['cmd_pos'] = msg
    
    def command_joint_vel_callback(self, msg):
        self.joint_data_msgs['cmd_vel'] = msg

    def publish_joint_states(self):
        """Publish current and commanded joint states for position, velocity, and torque."""
        for attr in ['pos', 'vel', 'torque']:
            current_msg = self.joint_data_msgs[attr]
            cmd_msg = self.joint_data_msgs[f'cmd_{attr}']
            
            if current_msg:
                for i, joint_name in enumerate(self.joint_name_list):
                    msg = Float32(data=current_msg.data[i])
                    self.joint_state_pubs[attr]['current'][joint_name].publish(msg)
            
            if cmd_msg:
                for i, joint_name in enumerate(self.joint_name_list):
                    msg = Float32(data=cmd_msg.data[i])
                    self.joint_state_pubs[attr]['command'][joint_name].publish(msg)

    def publish_limits(self):
        """Publish the joint limits for each joint (lower/upper limits)."""
        for attr, limits in self.joint_limits.items():
            lower_limit, upper_limit = limits
            if lower_limit and upper_limit:
                # Publish individual joint limits
                for i, joint_name in enumerate(self.joint_name_list):
                    # Publish lower limit
                    lower_msg = Float32(data=lower_limit[i])
                    self.joint_limit_pubs[attr]['lower'][joint_name].publish(lower_msg)

                    # Publish upper limit
                    upper_msg = Float32(data=upper_limit[i])
                    self.joint_limit_pubs[attr]['upper'][joint_name].publish(upper_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1-2.yaml', help='config file')
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Initialize the ROS2 node
    rclpy.init(args=None)
    node = rclpy.create_node('dof_vis_node')
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    rate = node.create_rate(10)  # 10 Hz

    # Initialize the VisJoint class
    vis_dof = VisJoint(node, config)
    cnt = 0
    start_time = time.time()
    try:
        while rclpy.ok():
            rate.sleep()
            vis_dof.publish_joint_states()
            vis_dof.publish_limits()  # Publish the joint limits (lower/upper)
            cnt += 1
            if cnt % 20 == 0:
                node.get_logger().info(f"FPS: {cnt / (time.time() - start_time)}")
    except KeyboardInterrupt:
        pass




