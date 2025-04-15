import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.ppo.ppo_new import PPO
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import time
import os
import statistics
from collections import deque
from hydra.utils import instantiate
from loguru import logger
from rich.progress import track
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
console = Console()

class PPOSym(PPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu',
                 num_replica=2):
        super().__init__(env, config, log_dir, device)
        self.transition_sym = RolloutStorage.Transition()
        self.num_replica = num_replica
        self.sym_loss_scale = self.config.sym_loss_scale
        
        # symmetrical dofs
        self.lower_left_dofs_idx_no = torch.tensor(self.env.config.robot.symmetric_dofs_idx.lower_left_dofs_idx_no).to(self.device)
        self.lower_right_dofs_idx_no = torch.tensor(self.env.config.robot.symmetric_dofs_idx.lower_right_dofs_idx_no).to(self.device)
        self.lower_left_dofs_idx_op = torch.tensor(self.env.config.robot.symmetric_dofs_idx.lower_left_dofs_idx_op).to(self.device)
        self.lower_right_dofs_idx_op = torch.tensor(self.env.config.robot.symmetric_dofs_idx.lower_right_dofs_idx_op).to(self.device)
        self.upper_left_dofs_idx_no = torch.tensor(self.env.config.robot.symmetric_dofs_idx.upper_left_dofs_idx_no).to(self.device)
        self.upper_right_dofs_idx_no = torch.tensor(self.env.config.robot.symmetric_dofs_idx.upper_right_dofs_idx_no).to(self.device)
        self.upper_left_dofs_idx_op = torch.tensor(self.env.config.robot.symmetric_dofs_idx.upper_left_dofs_idx_op).to(self.device)
        self.upper_right_dofs_idx_op = torch.tensor(self.env.config.robot.symmetric_dofs_idx.upper_right_dofs_idx_op).to(self.device)
        self.left_dofs_idx_no = torch.tensor(self.env.config.robot.symmetric_dofs_idx.left_dofs_idx_no).to(self.device)
        self.right_dofs_idx_no = torch.tensor(self.env.config.robot.symmetric_dofs_idx.right_dofs_idx_no).to(self.device)
        self.left_dofs_idx_op = torch.tensor(self.env.config.robot.symmetric_dofs_idx.left_dofs_idx_op).to(self.device)
        self.right_dofs_idx_op = torch.tensor(self.env.config.robot.symmetric_dofs_idx.right_dofs_idx_op).to(self.device)
        

    def _init_loss_dict_at_training_step(self):
        loss_dict = {}
        loss_dict['Value'] = 0
        loss_dict['Surrogate'] = 0
        loss_dict['Entropy'] = 0
        loss_dict["KL"] = 0
        loss_dict["Actor_Symmetric_Loss"] = 0
        loss_dict["Critic_Symmetric_Loss"] = 0
        return loss_dict

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs * self.num_replica, self.num_steps_per_env)
        ## Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            history_len = self.algo_history_length_dict.get(obs_key, 1)
            self.storage.register_key(obs_key, shape=(obs_dim * history_len,), dtype=torch.float)
        
        ## Register others
        self.storage.register_key('actions', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('rewards', shape=(1,), dtype=torch.float)
        self.storage.register_key('dones', shape=(1,), dtype=torch.bool)
        self.storage.register_key('values', shape=(1,), dtype=torch.float)
        self.storage.register_key('returns', shape=(1,), dtype=torch.float)
        self.storage.register_key('advantages', shape=(1,), dtype=torch.float)
        self.storage.register_key('actions_log_prob', shape=(1,), dtype=torch.float)
        self.storage.register_key('action_mean', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('action_sigma', shape=(self.num_act,), dtype=torch.float)

    def _actor_critic_rollout_step(self, obs_dict):
        self.transition.actions = self.actor.act(obs_dict["actor_obs"]).detach()
        self.transition.values = self.critic.evaluate(obs_dict["critic_obs"]).detach()
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions).detach().unsqueeze(1)
        self.transition.action_mean = self.actor.action_mean.detach()
        self.transition.action_sigma = self.actor.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.actor_obs = obs_dict["actor_obs"]
        self.transition.critic_obs = obs_dict["critic_obs"]
        
        actor_obs_sym = self._get_symmetirc_actor_obs(obs_dict["actor_obs"])
        critic_obs_sym = self._get_symmetirc_critic_obs(obs_dict["critic_obs"])
        self.transition_sym.actions = self.actor.act(actor_obs_sym).detach()
        self.transition_sym.values = self.critic.evaluate(critic_obs_sym).detach()
        self.transition_sym.actions_log_prob = self.actor.get_actions_log_prob(self.transition_sym.actions).detach().unsqueeze(1)
        self.transition_sym.action_mean = self.actor.action_mean.detach()
        self.transition_sym.action_sigma = self.actor.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition_sym.actor_obs = actor_obs_sym
        self.transition_sym.critic_obs = critic_obs_sym

        return self.transition.actions
    
    def _update_ppo(self, policy_state_dict, loss_dict):
        actor_loss, critic_loss, value_loss, surrogate_loss, entropy_loss, kl_mean = self._compute_ppo_loss(policy_state_dict)
        
        # Compute symmetric loss
        actor_obs = policy_state_dict["actor_obs"]
        critic_obs = policy_state_dict["critic_obs"]
        actor_obs_sym = self._get_symmetirc_actor_obs(actor_obs).detach()
        critic_obs_sym = self._get_symmetirc_critic_obs(critic_obs).detach()
        
        # import ipdb; ipdb.set_trace()
        actor_sym_loss = self.sym_loss_scale * torch.mean(torch.sum(torch.square(self.actor.act_inference(actor_obs_sym) - 
                                                                                 self._get_symmetric_actions(self.actor.act_inference(actor_obs)).detach()), dim=-1))
        critic_sym_loss = self.sym_loss_scale * torch.mean(torch.square(self.critic.evaluate(critic_obs_sym) - 
                                                                        self.critic.evaluate(critic_obs).detach()))
        actor_loss += actor_sym_loss
        critic_loss += critic_sym_loss
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        # Gradient step
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        loss_dict["KL"] = kl_mean.item()
        loss_dict["Actor_Symmetric_Loss"] += actor_sym_loss.item()
        loss_dict["Critic_Symmetric_Loss"] += critic_sym_loss.item() 
        return loss_dict

    def _process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards
        self.transition.dones = dones
        
        self.transition_sym.rewards = rewards
        self.transition_sym.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device)
            self.transition_sym.rewards += self.gamma * self.transition_sym.values * infos['time_outs'].unsqueeze(1).to(self.device)
        # Record the transition
        self.storage.add_transitions(self.transition, self.transition_sym)
        
        self.transition.clear()
        self.transition_sym.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)
    
    def _compute_returns(self, last_obs_dict, policy_state_dict):
        """Compute the returns and advantages for the given policy state.
        This function calculates the returns and advantages for each step in the 
        environment based on the provided observations and policy state. It uses 
        Generalized Advantage Estimation (GAE) to compute the advantages, which 
        helps in reducing the variance of the policy gradient estimates.
        Args:
            last_obs_dict (dict): The last observation dictionary containing the 
                      final state of the environment.
            policy_state_dict (dict): A dictionary containing the policy state 
                          information, including 'values', 'dones', 
                          and 'rewards'.
        Returns:
            tuple: A tuple containing:
            - returns (torch.Tensor): The computed returns for each step.
            - advantages (torch.Tensor): The normalized advantages for each step.
        """
        last_values = self.critic.evaluate(last_obs_dict["critic_obs"]).detach()
        last_values_sym = self.critic.evaluate(self._get_symmetirc_critic_obs(last_obs_dict["critic_obs"])).detach()
        last_values = torch.concatenate([last_values, last_values_sym], dim=0)
        advantage = 0
        
        values = policy_state_dict['values']
        dones = policy_state_dict['dones']
        rewards = policy_state_dict['rewards']
        
        last_values = last_values.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        rewards = rewards.to(self.device)
        
        returns = torch.zeros_like(values)
        
        num_steps = returns.shape[0]
        
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]

        # Compute and normalize the advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages
    
    def _get_symmetirc_actor_obs(self, actor_obs):
        """
        actor_obs: (batch_size, obs_dim * history_len)
        - keys:
            - 'base_ang_vel'
            - 'projected_gravity'
            - 'command_lin_vel'
            - 'command_ang_vel'
            - 'command_stand'
            - 'command_base_height'
            - 'ref_upper_dof_pos'
            - 'dof_pos'
            - 'dof_vel'
            - 'actions'
            - 'sin_phase'
            - 'cos_phase'
        """
        actor_obs = torch.clone(actor_obs[:, :(self.env.dim_obs["actor_obs"]*self.env.history_length["actor_obs"])])
        actor_obs = actor_obs.view(-1, self.env.history_length["actor_obs"], self.env.dim_obs["actor_obs"])
        actor_obs_sym = torch.zeros_like(actor_obs)
        # base_ang_vel
        actor_obs_sym[:, :, 0] = -actor_obs[:, :, 0]
        actor_obs_sym[:, :, 1] = actor_obs[:, :, 1]
        actor_obs_sym[:, :, 2] = -actor_obs[:, :, 2]
        # projected_gravity
        actor_obs_sym[:, :, 3] = actor_obs[:, :, 3]
        actor_obs_sym[:, :, 4] = -actor_obs[:, :, 4]
        actor_obs_sym[:, :, 5] = actor_obs[:, :, 5]
        # command_lin_vel
        actor_obs_sym[:, :, 6] = actor_obs[:, :, 6]
        actor_obs_sym[:, :, 7] = -actor_obs[:, :, 7]
        # command_ang_vel
        actor_obs_sym[:, :, 8] = -actor_obs[:, :, 8]
        # command_stand & command_base_height
        actor_obs_sym[:, :, 9] = actor_obs[:, :, 9]
        actor_obs_sym[:, :, 10] = actor_obs[:, :, 10]
        # ref_upper_dof_pos
        actor_obs_sym[:, :, 11 + self.upper_left_dofs_idx_no] = \
            actor_obs[:, :, 11 + self.upper_right_dofs_idx_no]
        actor_obs_sym[:, :, 11 + self.upper_right_dofs_idx_no] = \
            actor_obs[:, :, 11 + self.upper_left_dofs_idx_no]
        actor_obs_sym[:, :, 11 + self.upper_left_dofs_idx_op] = \
            -actor_obs[:, :, 11 + self.upper_right_dofs_idx_op]
        actor_obs_sym[:, :, 11 + self.upper_right_dofs_idx_op] = \
            -actor_obs[:, :, 11 + self.upper_left_dofs_idx_op]
        # dof_pos
        actor_obs_sym[:, :, 28 + self.left_dofs_idx_no] = \
            actor_obs[:, :, 28 + self.right_dofs_idx_no]
        actor_obs_sym[:, :, 28 + self.right_dofs_idx_no] = \
            actor_obs[:, :, 28 + self.left_dofs_idx_no]
        actor_obs_sym[:, :, 28 + self.left_dofs_idx_op] = \
            -actor_obs[:, :, 28 + self.right_dofs_idx_op]
        actor_obs_sym[:, :, 28 + self.right_dofs_idx_op] = \
            -actor_obs[:, :, 28 + self.left_dofs_idx_op]
        # dof_vel
        actor_obs_sym[:, :, 57 + self.left_dofs_idx_no] = \
            actor_obs[:, :, 57 + self.right_dofs_idx_no]
        actor_obs_sym[:, :, 57 + self.right_dofs_idx_no] = \
            actor_obs[:, :, 57 + self.left_dofs_idx_no]
        actor_obs_sym[:, :, 57 + self.left_dofs_idx_op] = \
            -actor_obs[:, :, 57 + self.right_dofs_idx_op]
        actor_obs_sym[:, :, 57 + self.right_dofs_idx_op] = \
            -actor_obs[:, :, 57 + self.left_dofs_idx_op]
        # actions
        actor_obs_sym[:, :, 86 + self.lower_left_dofs_idx_no] = \
            actor_obs[:, :, 86 + self.lower_right_dofs_idx_no]
        actor_obs_sym[:, :, 86 + self.lower_right_dofs_idx_no] = \
            actor_obs[:, :, 86 + self.lower_left_dofs_idx_no]
        actor_obs_sym[:, :, 86 + self.lower_left_dofs_idx_op] = \
            -actor_obs[:, :, 86 + self.lower_right_dofs_idx_op]
        actor_obs_sym[:, :, 86 + self.lower_right_dofs_idx_op] = \
            -actor_obs[:, :, 86 + self.lower_left_dofs_idx_op]
        # sin_phase & cos_phase
        actor_obs_sym[:, :, 98] = -actor_obs[:, :, 98]
        actor_obs_sym[:, :, 99] = -actor_obs[:, :, 99]
        # import ipdb; ipdb.set_trace()
        return actor_obs_sym.view(-1, self.env.dim_obs["actor_obs"]*self.env.history_length["actor_obs"])
    
    def _get_symmetirc_critic_obs(self, critic_obs):
        """
        critic_obs: (batch_size, obs_dim * history_len)
        - keys:
            - 'base_lin_vel'
            - 'base_ang_vel'
            - 'projected_gravity'
            - 'command_lin_vel'
            - 'command_ang_vel'
            - 'command_stand'
            - 'command_base_height'
            - 'ref_upper_dof_pos'
            - 'dof_pos'
            - 'dof_vel'
            - 'actions'
            - 'sin_phase'
            - 'cos_phase'
        """
        critic_obs = torch.clone(critic_obs[:, :(self.env.dim_obs["critic_obs"]*self.env.history_length["critic_obs"])])
        critic_obs = critic_obs.view(-1, self.env.history_length["critic_obs"], self.env.dim_obs["critic_obs"])
        critic_obs_sym = torch.zeros_like(critic_obs)
        # base_lin_vel
        critic_obs_sym[:, :, 0] = critic_obs[:, :, 0]
        critic_obs_sym[:, :, 1] = -critic_obs[:, :, 1]
        critic_obs_sym[:, :, 2] = critic_obs[:, :, 2]
        # base_ang_vel
        critic_obs_sym[:, :, 3] = -critic_obs[:, :, 3]
        critic_obs_sym[:, :, 4] = critic_obs[:, :, 4]
        critic_obs_sym[:, :, 5] = -critic_obs[:, :, 5]
        # projected_gravity
        critic_obs_sym[:, :, 6] = critic_obs[:, :, 6]
        critic_obs_sym[:, :, 7] = -critic_obs[:, :, 7]
        critic_obs_sym[:, :, 8] = critic_obs[:, :, 8]
        # command_lin_vel
        critic_obs_sym[:, :, 9] = critic_obs[:, :, 9]
        critic_obs_sym[:, :, 10] = -critic_obs[:, :, 10]
        # command_ang_vel
        critic_obs_sym[:, :, 11] = -critic_obs[:, :, 11]
        # command_stand & command_base_height
        critic_obs_sym[:, :, 12] = critic_obs[:, :, 12]
        critic_obs_sym[:, :, 13] = critic_obs[:, :, 13]
        # ref_upper_dof_pos
        critic_obs_sym[:, :, 14 + self.upper_left_dofs_idx_no] = \
            critic_obs[:, :, 14 + self.upper_right_dofs_idx_no]
        critic_obs_sym[:, :, 14 + self.upper_right_dofs_idx_no] = \
            critic_obs[:, :, 14 + self.upper_left_dofs_idx_no]
        critic_obs_sym[:, :, 14 + self.upper_left_dofs_idx_op] = \
            -critic_obs[:, :, 14 + self.upper_right_dofs_idx_op]
        critic_obs_sym[:, :, 14 + self.upper_right_dofs_idx_op] = \
            -critic_obs[:, :, 14 + self.upper_left_dofs_idx_op]
        # dof_pos
        critic_obs_sym[:, :, 31 + self.left_dofs_idx_no] = \
            critic_obs[:, :, 31 + self.right_dofs_idx_no]
        critic_obs_sym[:, :, 31 + self.right_dofs_idx_no] = \
            critic_obs[:, :, 31 + self.left_dofs_idx_no]
        critic_obs_sym[:, :, 31 + self.left_dofs_idx_op] = \
            -critic_obs[:, :, 31 + self.right_dofs_idx_op]
        critic_obs_sym[:, :, 31 + self.right_dofs_idx_op] = \
            -critic_obs[:, :, 31 + self.left_dofs_idx_op]
        # dof_vel
        critic_obs_sym[:, :, 60 + self.left_dofs_idx_no] = \
            critic_obs[:, :, 60 + self.right_dofs_idx_no]
        critic_obs_sym[:, :, 60 + self.right_dofs_idx_no] = \
            critic_obs[:, :, 60 + self.left_dofs_idx_no]
        critic_obs_sym[:, :, 60 + self.left_dofs_idx_op] = \
            -critic_obs[:, :, 60 + self.right_dofs_idx_op]
        critic_obs_sym[:, :, 60 + self.right_dofs_idx_op] = \
            -critic_obs[:, :, 60 + self.left_dofs_idx_op]
        # actions
        critic_obs_sym[:, :, 89 + self.lower_left_dofs_idx_no] = \
            critic_obs[:, :, 89 + self.lower_right_dofs_idx_no]
        critic_obs_sym[:, :, 89 + self.lower_right_dofs_idx_no] = \
            critic_obs[:, :, 89 + self.lower_left_dofs_idx_no]
        critic_obs_sym[:, :, 89 + self.lower_left_dofs_idx_op] = \
            -critic_obs[:, :, 89 + self.lower_right_dofs_idx_op]
        critic_obs_sym[:, :, 89 + self.lower_right_dofs_idx_op] = \
            -critic_obs[:, :, 89 + self.lower_left_dofs_idx_op]
        # sin_phase & cos_phase
        critic_obs_sym[:, :, 101] = -critic_obs[:, :, 101]
        critic_obs_sym[:, :, 102] = -critic_obs[:, :, 102]
        
        return critic_obs_sym.view(-1, self.env.dim_obs["critic_obs"]*self.env.history_length["critic_obs"])
    
    def _get_symmetric_actions(self, actions):
        """
        actions: (batch_size, num_act)
        """
        actions_sym = torch.zeros_like(actions)
        # lower body actions
        actions_sym[:, self.lower_left_dofs_idx_no] = \
            actions[:, self.lower_right_dofs_idx_no]
        actions_sym[:, self.lower_right_dofs_idx_no] = \
            actions[:, self.lower_left_dofs_idx_no]
        actions_sym[:, self.lower_left_dofs_idx_op] = \
            -actions[:, self.lower_right_dofs_idx_op]
        actions_sym[:, self.lower_right_dofs_idx_op] = \
            -actions[:, self.lower_left_dofs_idx_op]
        
        return actions_sym