import torch
from torch import Tensor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import json
from flask import send_file

from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.agents.ppo.ppo import PPO

from humanoidverse.utils.common import unnormalize

def quat_rotate_inverse_numpy(q, v):
    shape = q.shape
    # q_w corresponds to the scalar part of the quaternion
    q_w = q[:, -1]
    # q_vec corresponds to the vector part of the quaternion
    q_vec = q[:, 0:3]

    # Calculate a
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]

    # Calculate b
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0

    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return a - b + c

class AnalysisPlotDecoupledForceEstimator(RL_EvalCallback):
    training_loop: PPO
    env: LeggedRobotBase

    def __init__(self, config, training_loop: PPO):
        super().__init__(config, training_loop)
        env: LeggedRobotBase = self.training_loop.env
        self.env = env
        self.policy = self.training_loop._get_inference_policy()
        self.num_envs = self.env.num_envs
        self.logger = WebLogger(self.config.sim_dt)
        self.reset_buffers()
        self.log_single_robot = self.config.log_single_robot

        force_scale = self.env.config.obs.obs_scales.left_ee_apply_force # Or self.env.config.obs.obs_scales.right_ee_apply_force)
        self.force_min_x, self.force_max_x = self.env.config.apply_force_x_range[0], self.env.config.apply_force_x_range[1]
        self.force_min_y, self.force_max_y = self.env.config.apply_force_y_range[0], self.env.config.apply_force_y_range[1]
        self.force_min_z, self.force_max_z = self.env.config.apply_force_z_range[0], self.env.config.apply_force_z_range[1]
        self.force_min = np.array([self.force_min_x, self.force_min_y, self.force_min_z])
        self.force_max = np.array([self.force_max_x, self.force_max_y, self.force_max_z])

    def reset_buffers(self):
        self.obs_buf = [[] for _ in range(self.num_envs)]
        self.critic_obs_buf = [[] for _ in range(self.num_envs)]
        self.act_buf = [[] for _ in range(self.num_envs)]

    def on_pre_evaluate_policy(self):
        # Doing this in two lines because of type annotation issues.
        self.robot_num_dofs = self.env.num_dofs
        self.log_dof_pos_limits = self.env.dof_pos_limits.cpu().numpy()
        self.log_dof_vel_limits = self.env.dof_vel_limits.cpu().numpy()
        self.log_dof_torque_limits = self.env.torque_limits.cpu().numpy()
        self.logger.set_robot_limits(self.log_dof_pos_limits, self.log_dof_vel_limits, self.log_dof_torque_limits)
        self.logger.set_robot_num_dofs(self.robot_num_dofs)

    def on_post_evaluate_policy(self):
        pass

    def on_pre_eval_env_step(self, actor_state):
        obs: Tensor = actor_state["obs"]["actor_obs"].cpu()
        critic_obs: Tensor = actor_state["obs"]["critic_obs"].cpu()
        actions: Tensor = actor_state["actions"].cpu()
        # Unnormalize the estimated/apply forces
        # left_force_estimated = actor_state["left_ee_force_estimator_output"].cpu()
        # right_force_estimated = actor_state["right_ee_force_estimator_output"].cpu()
        left_force_estimated = unnormalize(actor_state["left_ee_force_estimator_output"].cpu().numpy(), 
                                           self.force_min, self.force_max)
        right_force_estimated = unnormalize(actor_state["right_ee_force_estimator_output"].cpu().numpy(), 
                                            self.force_min, self.force_max)
        left_force_apply = self.env.apply_force_tensor[:, self.env.left_hand_link_index, :].cpu().numpy()
        right_force_apply = self.env.apply_force_tensor[:, self.env.right_hand_link_index, :].cpu().numpy()
        base_quat = self.env.base_quat.cpu().numpy()
        left_force_apply = quat_rotate_inverse_numpy(base_quat, left_force_apply)
        right_force_apply = quat_rotate_inverse_numpy(base_quat, right_force_apply)

        for i in range(self.num_envs):
            self.obs_buf[i].append(obs[i])
            self.critic_obs_buf[i].append(critic_obs[i])
            self.act_buf[i].append(actions[i])


        # Note: we divide by 0.1 because the forces are scaled by 0.1 in the obs space
        if self.log_single_robot:
            self.logger.log_states(
                {
                # 'dof_pos_target': actions[0].cpu().numpy(),
                # 'dof_pos': self.env.simulator.dof_pos[0].cpu().numpy(),
                # 'dof_vel': self.env.simulator.dof_vel[0].cpu().numpy(),
                # 'dof_torque': self.env.torques[0].cpu().numpy(),
                'base_vel_x': self.env.base_lin_vel[0, 0].item(),
                'base_vel_y': self.env.base_lin_vel[0, 1].item(),
                'base_vel_z': self.env.base_lin_vel[0, 2].item(),
                'base_vel_yaw': self.env.base_ang_vel[0, 2].item(),
                'contact_forces_z': self.env.simulator.contact_forces[0, self.env.feet_indices, 2].cpu().numpy(),
                # 'left_ee_force': self.env.apply_force_tensor[0, self.env.left_hand_link_index, :].cpu().numpy(),
                # 'right_ee_force': self.env.apply_force_tensor[0, self.env.right_hand_link_index, :].cpu().numpy(),
                # 'left_force_estimated': left_force_estimated[0].cpu().numpy() / 0.1,
                # 'right_force_estimated': right_force_estimated[0].cpu().numpy() / 0.1,
                'left_ee_force': left_force_apply[0],
                'right_ee_force': right_force_apply[0],
                'left_force_estimated': left_force_estimated[0],
                'right_force_estimated': right_force_estimated[0],
                }
            )
        else:
            # log average of all robots
            self.logger.log_states(
                {
                    # 'dof_pos_target': actions.mean(dim=0).cpu().numpy(),
                    # 'dof_pos': self.env.simulator.dof_pos.mean(dim=0).cpu().numpy(),
                    # 'dof_vel': self.env.simulator.dof_vel.mean(dim=0).cpu().numpy(),
                    # 'dof_torque': self.env.torques.mean(dim=0).cpu().numpy(),
                    'base_vel_x': self.env.base_lin_vel[:, 0].mean().item(),
                    'base_vel_y': self.env.base_lin_vel[:, 1].mean().item(),
                    'base_vel_z': self.env.base_lin_vel[:, 2].mean().item(),
                    'base_vel_yaw': self.env.base_ang_vel[:, 2].mean().item(),
                    'contact_forces_z': self.env.simulator.contact_forces[:, self.env.feet_indices, 2].mean(dim=1).cpu().numpy(),
                    # 'left_ee_force': self.env.apply_force_tensor[:, self.env.left_hand_link_index, :].mean(dim=0).cpu().numpy(),
                    # 'right_ee_force': self.env.apply_force_tensor[:, self.env.right_hand_link_index, :].mean(dim=0).cpu().numpy(),
                    # 'left_force_estimated': left_force_estimated.mean(dim=0).cpu().numpy() / 0.1,
                    # 'right_force_estimated': right_force_estimated.mean(dim=0).cpu().numpy() / 0.1,
                    'left_ee_force': left_force_apply.mean(dim=0),
                    'right_ee_force': right_force_apply.mean(dim=0),
                    'left_force_estimated': left_force_estimated.mean(dim=0),
                    'right_force_estimated': right_force_estimated.mean(dim=0),
                }
            )
        return actor_state

    def on_post_eval_env_step(self, actor_state):
        step = actor_state["step"]
        if (step + 1) % self.config.plot_update_interval == 0:
            self.logger.plot_states()
        return actor_state

class WebLogger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.thread = None

    def set_robot_limits(self, dof_pos_limits, dof_vel_limits, dof_torque_limits):
        self.log_dof_pos_limits = dof_pos_limits
        self.log_dof_vel_limits = dof_vel_limits
        self.log_dof_torque_limits = dof_torque_limits

    def set_robot_num_dofs(self, num_dofs):
        self.robot_num_dofs = num_dofs

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run_server)
            self.thread.start()
        self._update_plot()
    
    def _run_server(self):
        @self.app.route('/')
        def index():
            return send_file('analysis_plot_template.html')

        self.socketio.run(self.app, debug=False, use_reloader=False)
    
    def _update_plot(self):
        log = self.state_log.copy()
        total_time = len(next(iter(log.values()))) * self.dt
        time = np.linspace(0, total_time, len(next(iter(log.values()))))

        for key in log:
            if isinstance(log[key], list):
                log[key] = np.array(log[key])

        BLUE = '#005A9D'
        RED = '#DA2513'
        YELLOW = '#D4A017'
        BLACK = '#000000'

        force_to_position_scale = 0.05 # force to position scale, for better visualization
        force_to_position_bias = 0.0 # bias to force to position, for better visualization

        num_dofs = self.robot_num_dofs

        def get_subplot_titles():
            titles = [
                'Left EE Force X',
                'Left EE Force Y',
                'Left EE Force Z',
                'Left EE Force',
                'Right EE Force X',
                'Right EE Force Y',
                'Right EE Force Z',
                'Right EE Force',
                'None',
                'None',
                'None',
                'Vertical Contact Force Z', 
                'Left EE Force Error X',
                'Left EE Force Error Y',
                'Left EE Force Error Z',
                'Left EE Force Error',
                'Right EE Force Error X',
                'Right EE Force Error Y',
                'Right EE Force Error Z',
                'Right EE Force Error',
            ]
            return titles

        # Calculate number of rows needed
        additional_rows = 6
        num_rows = additional_rows # + num_dofs

        fig = make_subplots(rows=num_rows, cols=4, subplot_titles=get_subplot_titles())

        def add_trace(x, y, color, row, col, name=None, show_legend=False):
            fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color), name=name, showlegend=show_legend), row=row, col=col)
        # print("log['left_ee_force'][:, 0]: ", log['left_ee_force'][:, 0])
        # print("log['left_ee_force'][:, 1]: ", log['left_ee_force'][:, 1])
        # print("log['left_ee_force'][:, 2]: ", log['left_ee_force'][:, 2])
        add_trace(time, log["left_ee_force"][:, 0], BLUE, 1, 1, "Left EE Force X")
        add_trace(time, log["left_ee_force"][:, 1], BLUE, 1, 2, "Left EE Force Y")
        add_trace(time, log["left_ee_force"][:, 2], BLUE, 1, 3, "Left EE Force Z")
        add_trace(time, np.linalg.norm(log["left_ee_force"], axis=1), RED, 1, 4, "Left EE Force")

        add_trace(time, log["right_ee_force"][:, 0], RED, 2, 1, "Right EE Force X")
        add_trace(time, log["right_ee_force"][:, 1], RED, 2, 2, "Right EE Force Y")
        add_trace(time, log["right_ee_force"][:, 2], RED, 2, 3, "Right EE Force Z")
        add_trace(time, np.linalg.norm(log["right_ee_force"], axis=1), RED, 2, 4, "Right EE Force")

        # Vertical Contact forces
        forces = log["contact_forces_z"]
        for i in range(forces[0].shape[0]):
            add_trace(time, [force[i] for force in forces], BLUE, 3, 4, f"Force {i}")

        # error

        add_trace(time, log["left_ee_force"][:, 0], RED, 4, 1, "Left EE Force X")
        add_trace(time, log["left_force_estimated"][:, 0], YELLOW, 4, 1, "Estimated Left EE Force X")
        add_trace(time, log["left_ee_force"][:, 1], RED, 4, 2, "Left EE Force Y")
        add_trace(time, log["left_force_estimated"][:, 1], YELLOW, 4, 2, "Estimated Left EE Force Y")
        add_trace(time, log["left_ee_force"][:, 2], RED, 4, 3, "Left EE Force Z")
        add_trace(time, log["left_force_estimated"][:, 2], YELLOW, 4, 3, "Estimated Left EE Force Z")
        add_trace(time, np.linalg.norm(log["left_ee_force"], axis=1), RED, 4, 4, "Left EE Force")
        add_trace(time, np.linalg.norm(log["left_force_estimated"], axis=1), YELLOW, 4, 4, "Estimated Left EE Force")

        add_trace(time, log["right_ee_force"][:, 0], RED, 5, 1, "Right EE Force X")
        add_trace(time, log["right_force_estimated"][:, 0], YELLOW, 5, 1, "Estimated Right EE Force X")
        add_trace(time, log["right_ee_force"][:, 1], RED, 5, 2, "Right EE Force Y")
        add_trace(time, log["right_force_estimated"][:, 1], YELLOW, 5, 2, "Estimated Right EE Force Y")
        add_trace(time, log["right_ee_force"][:, 2], RED, 5, 3, "Right EE Force Z")
        add_trace(time, log["right_force_estimated"][:, 2], YELLOW, 5, 3, "Estimated Right EE Force Z")
        add_trace(time, np.linalg.norm(log["right_ee_force"], axis=1), RED, 5, 4, "Right EE Force")
        add_trace(time, np.linalg.norm(log["right_force_estimated"], axis=1), YELLOW, 5, 4, "Estimated Right EE Force")

        fig.update_layout(height=300*num_rows, width=1500, title_text="Robot State Plots", showlegend=True)
        
        # Update x and y axis labels
        for i in range(num_rows):
            for j in range(3):
                fig.update_xaxes(title_text="time [s]", row=i+1, col=j+1)
            fig.update_xaxes(title_text="", row=i+1, col=4)

        
        
        fig.update_yaxes(title_text="Left EE Force X [N]", row=1, col=1)
        fig.update_yaxes(title_text="Left EE Force Y [N]", row=1, col=2)
        fig.update_yaxes(title_text="Left EE Force Z [N]", row=1, col=3)
        fig.update_yaxes(title_text="Left EE Force [N]", row=1, col=4)
        fig.update_yaxes(title_text="Right EE Force X [N]", row=2, col=1)
        fig.update_yaxes(title_text="Right EE Force Y [N]", row=2, col=2)
        fig.update_yaxes(title_text="Right EE Force Z [N]", row=2, col=3)
        fig.update_yaxes(title_text="Right EE Force [N]", row=2, col=4)

        fig.update_yaxes(title_text="Vertical Contact Forces [N]", row=3, col=4)

        fig.update_yaxes(title_text="Left EE Force Error X [N]", row=4, col=1)
        fig.update_yaxes(title_text="Left EE Force Error Y [N]", row=4, col=2)
        fig.update_yaxes(title_text="Left EE Force Error Z [N]", row=4, col=3)
        fig.update_yaxes(title_text="Left EE Force Error [N]", row=4, col=4)
        fig.update_yaxes(title_text="Right EE Force Error X [N]", row=5, col=1)
        fig.update_yaxes(title_text="Right EE Force Error Y [N]", row=5, col=2)
        fig.update_yaxes(title_text="Right EE Force Error Z [N]", row=5, col=3)
        fig.update_yaxes(title_text="Right EE Force Error [N]", row=5, col=4)

        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        self.socketio.emit('update_plots', plot_json)

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.thread:
            self.socketio.stop()
            self.thread.join()