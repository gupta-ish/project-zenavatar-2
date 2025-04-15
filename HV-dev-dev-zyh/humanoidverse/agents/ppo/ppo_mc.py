import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict
from humanoidverse.agents.ppo.ppo_new import PPO

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

class PPOMultiCritic(PPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        super().__init__(env, config, log_dir, device)

    def _init_config(self):
        super()._init_config()
        # Multi-critic reward related Config
        self.rw_groups = self.env.config.reward.reward_groups
        self.rw_weights = self.env.config.reward.reward_weights

    def _setup_models_and_optimizer(self):
        self.actor = PPOActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std
        ).to(self.device)

        self.critics = {}
        self.critic_keys = self.env.config.reward.reward_groups.keys()
        for critic_key in self.critic_keys:
            self.critics[critic_key] = PPOCritic(self.algo_obs_dim_dict,
                                                 self.config.module_dict.critic).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizers = {}
        for critic_key in self.critic_keys:
            self.critic_optimizers[critic_key] = optim.Adam(self.critics[critic_key].parameters(), lr=self.critic_learning_rate)

    def _eval_mode(self):
        self.actor.eval()
        for critic in self.critics.values():
            critic.eval()

    def _train_mode(self):
        self.actor.train()
        for critic in self.critics.values():
            critic.train()

    def load(self, ckpt_path):
        # import ipdb; ipdb.set_trace()
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            for critic_key in self.critic_keys:
                self.critics[critic_key].load_state_dict(loaded_dict["critic_model_state_dict"][critic_key])
            if self.load_optimizer:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                for critic_key in self.critic_keys:
                    self.critic_optimizers[critic_key].load_state_dict(loaded_dict["critic_optimizer_state_dict"][critic_key])
                self.actor_learning_rate = loaded_dict['actor_optimizer_state_dict']['param_groups'][0]['lr']
                self.critic_learning_rate = loaded_dict['critic_optimizer_state_dict']['param_groups'][0]['lr']
                self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate)
                logger.info(f"Optimizer loaded from checkpoint")
                logger.info(f"Actor Learning rate: {self.actor_learning_rate}")
                logger.info(f"Critic Learning rate: {self.critic_learning_rate}")
            self.current_learning_iteration = loaded_dict["iter"]
            return loaded_dict["infos"]

    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        torch.save({
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critics,
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizers,
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def _multi_critic_evaluation(self, critic_obs):
        critic_values = {}
        for critic_key, critic in self.critics.items():
            critic_values[critic_key] = critic.evaluate(critic_obs)
        return critic_values

    def _actor_critic_rollout_step(self, obs_dict):
        self.transition.actions = self.actor.act(obs_dict["actor_obs"]).detach()
        self.transition.values = self._multi_critic_evaluation(obs_dict["critic_obs"]).detach()
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions).detach().unsqueeze(1)
        self.transition.action_mean = self.actor.action_mean.detach()
        self.transition.action_sigma = self.actor.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.actor_obs = obs_dict["actor_obs"]
        self.transition.critic_obs = obs_dict["critic_obs"]
        
        assert len(self.transition.actions.shape) == 2
        assert len(self.transition.actions_log_prob.shape) == 2
        assert len(self.transition.action_mean.shape) == 2
        assert len(self.transition.action_sigma.shape) == 2

        return self.transition.actions

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # Compute the actions and values
                actions = self._actor_critic_rollout_step(obs_dict)
                actor_state = {}
                actor_state["actions"] = actions
                obs_dict, rewards, dones, infos = self.env.step(actor_state)

                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                
                dones = dones.to(self.device)
                rewards_weighted = torch.zeros_like(dones, dtype=torch.float)
                for rw_key, rw_value in rewards.items():
                    # TODO [Yuanhang]: Add RunningMeanStd for rewards
                    rewards[rw_key] = rw_value.to(self.device)
                    rewards_weighted += rewards[rw_key] * self.rw_weights[rw_key]

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards_weighted.clone().unsqueeze(1)
                dones_stored = dones.clone().unsqueeze(1)

                self._process_env_step(rewards_stored, dones_stored, infos)

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time
            
            # prepare data for training
            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(values=self.storage.query_key('values'), 
                                       dones=self.storage.query_key('dones'), 
                                       rewards=self.storage.query_key('rewards'))
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict

    def _process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device)
        assert len(self.transition.rewards.shape) == 2
        # Record the transition
        self.storage.add_transitions(self.transition)
        
        self.transition.clear()
        self.actor.reset(dones)
        for critic in self.critics.values():
            critic.reset(dones)

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
        # last_values= self.critic.evaluate(last_obs_dict["critic_obs"]).detach()
        last_values = self._multi_critic_evaluation(last_obs_dict["critic_obs"]).detach()
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
    
    def _compute_ppo_loss(self, policy_state_dict):
        actions_batch = policy_state_dict['actions']
        # target_values_batch = policy_state_dict['returns'] # This is wrong
        target_values_batch = policy_state_dict['values'] # This is correct
        advantages_batch = policy_state_dict['advantages']
        returns_batch = policy_state_dict['returns']
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob']
        old_mu_batch = policy_state_dict['action_mean']
        old_sigma_batch = policy_state_dict['action_sigma']

        self.actor.act(policy_state_dict["actor_obs"])
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        # value_batch = self.critic.evaluate(policy_state_dict["critic_obs"])
        value_batch = self._multi_critic_evaluation(policy_state_dict["critic_obs"])
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy

        # KL
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + 
                    (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch) + 1.e-5) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)
                
                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                    self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                    self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)

                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.actor_learning_rate
                for critic_optimizer in self.critic_optimizers.values():
                    for param_group in critic_optimizer.param_groups:
                        param_group['lr'] = self.critic_learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                        1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                            self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        entropy_loss = entropy_batch.mean()
        actor_loss = surrogate_loss - self.entropy_coef * entropy_loss
        
        critic_loss = self.value_loss_coef * value_loss
        
        return actor_loss, critic_loss, value_loss, surrogate_loss, entropy_loss, kl_mean
    
    def _update_ppo(self, policy_state_dict, loss_dict):
        actor_loss, critic_loss, value_loss, surrogate_loss, entropy_loss, kl_mean = self._compute_ppo_loss(policy_state_dict)

        self.actor_optimizer.zero_grad()
        for critic_optimizer in self.critic_optimizers.values():
            critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        # Gradient step
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        for critic in self.critics.values():
            nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        for critic_optimizer in self.critic_optimizers.values():
            critic_optimizer.step()

        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        loss_dict["KL"] += kl_mean.item()
        return loss_dict


    @property
    def inference_model(self):
        return {
            "actor": self.actor,
            "critics": self.critics
        }

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################