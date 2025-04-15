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
        self.num_joints = len(config['dof_pos_lower_limit_list'])
        self.num_joints_vis = 5
        # Define maximum joint temperature
        self.max_joint_temp = 50.0
        # 
        self.publish_out_of_limit = False
        # Joint limits
        self.joint_limits = {
            'pos': (config['dof_pos_lower_limit_list'], config['dof_pos_upper_limit_list']),
            'vel': ([-lim for lim in config['dof_vel_limit_list']], config['dof_vel_limit_list']),
            'torque': ([-lim for lim in config['dof_effort_limit_list']], config['dof_effort_limit_list']),
            'temp': ([-100.0]*self.num_joints, [self.max_joint_temp] * self.num_joints),
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

        ############### Publishers for each joint's data (position, velocity, torque, temperature) ###############
        self.joint_data_pubs = {
            'pos': [self.node.create_publisher(Float32, f"robot_joint_pos_{i}", 1) for i in range(self.num_joints)],
            'vel': [self.node.create_publisher(Float32, f"robot_joint_vel_{i}", 1) for i in range(self.num_joints)],
            'torque': [self.node.create_publisher(Float32, f"robot_joint_torque_{i}", 1) for i in range(self.num_joints)],
            'temp': [self.node.create_publisher(Float32, f"robot_joint_temp_{i}", 1) for i in range(self.num_joints)],
        }
        self.joint_cmd_pubs = {
            'pos': [self.node.create_publisher(Float32, f"command_joint_pos_{i}", 1) for i in range(self.num_joints)],
            'vel': [self.node.create_publisher(Float32, f"command_joint_vel_{i}", 1) for i in range(self.num_joints)],
            'torque': [self.node.create_publisher(Float32, f"command_joint_torque_{i}", 1) for i in range(self.num_joints)],
        }
        # Individual publishers for joint position limits
        self.joint_pos_limit_pubs = {
            'lower': [self.node.create_publisher(Float32, f"/joint_pos_lower_limit_{i}", 1) for i in range(self.num_joints)],
            'upper': [self.node.create_publisher(Float32, f"/joint_pos_upper_limit_{i}", 1) for i in range(self.num_joints)],
        }
        self.joint_vel_limit_pubs = {
            'lower': [self.node.create_publisher(Float32, f"/joint_vel_lower_limit_{i}", 1) for i in range(self.num_joints)],
            'upper': [self.node.create_publisher(Float32, f"/joint_vel_upper_limit_{i}", 1) for i in range(self.num_joints)],
        }
        self.joint_torque_limit_pubs = {
            'lower': [self.node.create_publisher(Float32, f"/joint_torque_lower_limit_{i}", 1) for i in range(self.num_joints)],
            'upper': [self.node.create_publisher(Float32, f"/joint_torque_upper_limit_{i}", 1) for i in range(self.num_joints)],
        }
        self.joint_temp_limit_pubs = {
            'lower': [self.node.create_publisher(Float32, f"/joint_temp_lower_limit_{i}", 1) for i in range(self.num_joints)],
            'upper': [self.node.create_publisher(Float32, f"/joint_temp_upper_limit_{i}", 1) for i in range(self.num_joints)],
        }
        self.joint_all_limit_pubs = {
            'pos': self.joint_pos_limit_pubs,
            'vel': self.joint_vel_limit_pubs,
            'torque': self.joint_torque_limit_pubs,
            'temp': self.joint_temp_limit_pubs,
        }

        self.running = True  # Flag to control thread execution
        self.lock = threading.Lock()  # Lock for thread-safe access to shared data

        # Start threads for each attribute
        self.threads = []
        for attr in ['pos', 'vel', 'torque', 'temp']:
            thread = threading.Thread(target=self._publish_worker, args=(attr,))
            thread.start()
            self.threads.append(thread)

        ############### Publishers for out-of limit joints ###############
        self.out_of_limit_arrays = {
            'pos': np.zeros(self.num_joints, dtype=int),
            'vel': np.zeros(self.num_joints, dtype=int),
            'torque': np.zeros(self.num_joints, dtype=int),
            'temp': np.zeros(self.num_joints, dtype=int),
        }

        # Publishers for joint status arrays
        self.joint_status_pubs = {
            attr: self.node.create_publisher(Float64MultiArray, f"{attr}_out_of_limit_idx", 1)
            for attr in self.joint_limits
        }

        # Publishers for individual out-of-limit joints
        self.out_of_limit_pubs = {
            attr: [self.node.create_publisher(Float32, f"out_of_limit_{attr}_joint_{i}", 1) for i in range(self.num_joints_vis)]
            for attr in self.joint_limits
        }
        self.out_of_lower_limit_pubs = {
            attr: [self.node.create_publisher(Float32, f"out_of_lower_limit_{attr}_joint_{i}", 1) for i in range(self.num_joints_vis)]
            for attr in self.joint_limits
        }
        self.out_of_upper_limit_pubs = {
            attr: [self.node.create_publisher(Float32, f"out_of_upper_limit_{attr}_joint_{i}", 1) for i in range(self.num_joints_vis)]
            for attr in self.joint_limits
        }
        self.out_of_limit_cmd_pubs = {
            attr: [self.node.create_publisher(Float32, f"out_of_limit_{attr}_cmd_joint_{i}", 1) for i in range(self.num_joints_vis)]
            for attr in self.joint_limits
        }
        self.out_of_limits_idx = {
            attr: deque(maxlen=self.num_joints_vis) for attr in self.joint_limits
        }

        # Publishers for joint limits (lower and upper)
        self.joint_limit_pubs = {
            attr: {
                'lower': self.node.create_publisher(Float64MultiArray, f"/joint_{attr}_lower_limit", 1),
                'upper': self.node.create_publisher(Float64MultiArray, f"/joint_{attr}_upper_limit", 1),
            }
            for attr in self.joint_limits
        }

        ############### Subscribers for joint data ###############
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

    def update_out_of_limit_status(self):
        """Update the out-of-limit arrays for all attributes."""
        for attr, limits in self.joint_limits.items():
            msg = self.joint_data_msgs[attr]
            if msg:
                lower_limit, upper_limit = limits
                for i, value in enumerate(msg.data):
                    if (
                        (lower_limit is not None and value <= lower_limit[i])
                        or (upper_limit is not None and value >= upper_limit[i])
                    ):
                        self.out_of_limit_arrays[attr][i] = 1
                        if i not in self.out_of_limits_idx[attr]:
                            if len(self.out_of_limits_idx[attr]) < self.num_joints_vis:
                                print("attr: ", attr, "i: ", i)
                                self.out_of_limits_idx[attr].append(i)

    def publish_out_of_limit_joints(self):
        """Detect and publish the first 3 out-of-limit joints for each attribute."""
        for attr, limits in self.joint_limits.items():
            msg = self.joint_data_msgs[attr]
            if msg:
                lower_limit, upper_limit = limits
                if self.out_of_limits_idx[attr]:
                    for i, idx in enumerate(self.out_of_limits_idx[attr]):
                        pub = self.out_of_limit_pubs[attr][i]
                        lower_pub = self.out_of_lower_limit_pubs[attr][i]
                        upper_pub = self.out_of_upper_limit_pubs[attr][i]
                        cmd_pub = self.out_of_limit_cmd_pubs[attr][i]
                        pub.publish(Float32(data=msg.data[idx]))
                        lower_pub.publish(Float32(data=lower_limit[idx]))
                        upper_pub.publish(Float32(data=upper_limit[idx]))
                        cmd_pub.publish(Float32(data=self.joint_data_msgs[f'cmd_{attr}'].data[idx]))

    def publish_out_of_limit_status(self):
        """Publish the current status of the out-of-limit idx."""
        for attr, pub in self.joint_status_pubs.items():
            pub.publish(Float64MultiArray(data=list(self.out_of_limits_idx[attr])))

    def publish_limits(self):
        """Publish the joint limits for each joint (lower/upper limits)."""
        for attr, limits in self.joint_limits.items():
            lower_limit, upper_limit = limits
            if lower_limit and upper_limit:
                # Publish lower limit
                lower_pub = self.joint_limit_pubs[attr]['lower']
                lower_pub.publish(Float64MultiArray(data=lower_limit))

                # Publish upper limit
                upper_pub = self.joint_limit_pubs[attr]['upper']
                upper_pub.publish(Float64MultiArray(data=upper_limit))

    def _publish_worker(self, attr):
        """Worker thread to continuously publish data for a specific attribute."""
        while self.running:
            msg = None
            with self.lock:
                if self.joint_data_msgs[attr]:
                    msg = self.joint_data_msgs[attr]
                    if attr != 'temp':
                        cmd_msg = self.joint_data_msgs[f'cmd_{attr}']
                    else:
                        cmd_msg = True

            if msg and cmd_msg:
                for i in range(self.num_joints):
                    self.joint_data_pubs[attr][i].publish(Float32(data=msg.data[i]))
                    if attr != 'temp':
                        self.joint_cmd_pubs[attr][i].publish(Float32(data=cmd_msg.data[i]))
                    self.joint_all_limit_pubs[attr]['lower'][i].publish(Float32(data=self.joint_limits[attr][0][i]))
                    self.joint_all_limit_pubs[attr]['upper'][i].publish(Float32(data=self.joint_limits[attr][1][i]))

            # Add a short delay to prevent thread from consuming too much CPU
            time.sleep(0.01)

    def stop_threads(self):
        """Stop all worker threads."""
        self.running = False
        for thread in self.threads:
            thread.join()

    def update_joint_data(self, data, attr):
        """Update joint data for a specific attribute."""
        with self.lock:
            self.joint_data_msgs[attr] = data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot')
    parser.add_argument('--config', type=str, default='config/h1-2.yaml', help='config file')
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Initialize the ROS2 node
    rclpy.init(args=None)
    node = rclpy.create_node('limit_joint_vis_node')
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    rate = node.create_rate(300)  # 10 Hz

    # Initialize the VisJoint class
    vis_dof = VisJoint(node, config)
    cnt = 0
    start_time = time.time()
    try:
        while rclpy.ok():
            rate.sleep()
            if vis_dof.publish_out_of_limit:
                vis_dof.publish_out_of_limit_joints()
                vis_dof.update_out_of_limit_status()
                vis_dof.publish_out_of_limit_status()
            vis_dof.publish_limits()  # Publish the joint limits (lower/upper)
            cnt += 1
            if cnt % 20 == 0:
                node.get_logger().info(f"FPS: {cnt / (time.time() - start_time)}")
    except KeyboardInterrupt:
        pass
    finally:
        vis_dof.stop_threads()  # Stop worker threads gracefully