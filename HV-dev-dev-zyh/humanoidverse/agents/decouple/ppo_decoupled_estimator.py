import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.decouple.ppo_decoupled import PPODecoupled
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict
from humanoidverse.agents.modules.modules import BaseModule
import torch.nn.functional as F
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from humanoidverse.utils.common import normalize, unnormalize

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

class PPODecoupledEstimator(PPODecoupled):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        super().__init__(env, config, log_dir, device)
        self.use_apply_or_estimated_force = 0 # 0: apply force, 1: estimated force
        self.valid_estimated_force_envs = None 

    def _init_config(self):
        super()._init_config()
        self.num_act = self.env.config.robot.lower_body_actions_dim
        self.left_ee_estimator_learning_rate = self.config.estimator_learning_rate
        self.right_ee_estimator_learning_rate = self.config.estimator_learning_rate
        force_scale = self.env.config.obs.obs_scales.left_ee_apply_force # Or self.env.config.obs.obs_scales.right_ee_apply_force)
        self.force_min_x, self.force_max_x = self.env.config.apply_force_x_range[0] * force_scale, self.env.config.apply_force_x_range[1] * force_scale
        self.force_min_y, self.force_max_y = self.env.config.apply_force_y_range[0] * force_scale, self.env.config.apply_force_y_range[1] * force_scale
        self.force_min_z, self.force_max_z = self.env.config.apply_force_z_range[0] * force_scale, self.env.config.apply_force_z_range[1] * force_scale
        self.force_min = torch.tensor([self.force_min_x, self.force_min_y, self.force_min_z], device=self.device)
        self.force_max = torch.tensor([self.force_max_x, self.force_max_y, self.force_max_z], device=self.device)
    
    def _setup_models_and_optimizer(self):
        super()._setup_models_and_optimizer()
        # Yuanhang: the estimator outputs the normalized force
        self.left_ee_force_estimator = BaseModule(self.algo_obs_dim_dict,
                                                  self.config.module_dict.left_ee_force_estimator).to(self.device)
        self.right_ee_force_estimator = BaseModule(self.algo_obs_dim_dict,
                                                   self.config.module_dict.right_ee_force_estimator).to(self.device)
        self.left_ee_force_estimator_optimizer = optim.Adam(self.left_ee_force_estimator.parameters(),
                                                            lr=self.left_ee_estimator_learning_rate, 
                                                            weight_decay=self.config.weight_decay) # Add L2 regularization
        self.right_ee_force_estimator_optimizer = optim.Adam(self.right_ee_force_estimator.parameters(),
                                                             lr=self.right_ee_estimator_learning_rate, 
                                                             weight_decay=self.config.weight_decay) # Add L2 regularization

    def setup(self):
        logger.info("Setting up PPO_Decoupled Estimator")
        self._setup_models_and_optimizer()
        logger.info(f"Setting up Storage")
        self._setup_storage()
    
    def _setup_storage(self):
        super()._setup_storage()
        print(f"Algo obs dim dict: {self.algo_obs_dim_dict}")

    def set_learning_rate(self, actor_learning_rate, critic_learning_rate,
                          left_ee_estimator_learning_rate, right_ee_estimator_learning_rate):
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.left_ee_estimator_learning_rate = left_ee_estimator_learning_rate
        self.right_ee_estimator_learning_rate = right_ee_estimator_learning_rate

    def load(self, ckpt_path):
        # import ipdb; ipdb.set_trace()
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            self.left_ee_force_estimator.load_state_dict(loaded_dict["left_ee_force_estimator_state_dict"])
            self.right_ee_force_estimator.load_state_dict(loaded_dict["right_ee_force_estimator_state_dict"])
            if self.load_optimizer:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                self.left_ee_force_estimator_optimizer.load_state_dict(loaded_dict["left_ee_force_estimator_optimizer_state_dict"])
                self.right_ee_force_estimator_optimizer.load_state_dict(loaded_dict["right_ee_force_estimator_optimizer_state_dict"])
                self.actor_learning_rate = loaded_dict['actor_optimizer_state_dict']['param_groups'][0]['lr']
                self.critic_learning_rate = loaded_dict['critic_optimizer_state_dict']['param_groups'][0]['lr']
                self.left_ee_estimator_learning_rate = loaded_dict['left_ee_force_estimator_optimizer_state_dict']['param_groups'][0]['lr']
                self.right_ee_estimator_learning_rate = loaded_dict['right_ee_force_estimator_optimizer_state_dict']['param_groups'][0]['lr']

                self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate,
                                       self.left_ee_estimator_learning_rate, self.right_ee_estimator_learning_rate)
                logger.info(f"Optimizer loaded from checkpoint")
                logger.info(f"Actor Learning rate: {self.actor_learning_rate}")
                logger.info(f"Critic Learning rate: {self.critic_learning_rate}")
                logger.info(f"Left EE Estimator Learning rate: {self.left_ee_estimator_learning_rate}")
                logger.info(f"Right EE Estimator Learning rate: {self.right_ee_estimator_learning_rate}")
            self.current_learning_iteration = loaded_dict["iter"]
            return loaded_dict["infos"]

    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        torch.save({
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'left_ee_force_estimator_state_dict': self.left_ee_force_estimator.state_dict(),
            'right_ee_force_estimator_state_dict': self.right_ee_force_estimator.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'left_ee_force_estimator_optimizer_state_dict': self.left_ee_force_estimator_optimizer.state_dict(),
            'right_ee_force_estimator_optimizer_state_dict': self.right_ee_force_estimator_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def _eval_mode(self):
        super()._eval_mode()
        self.left_ee_force_estimator.eval()
        self.right_ee_force_estimator.eval()
    
    def _train_mode(self):
        super()._train_mode()
        self.left_ee_force_estimator.train()
        self.right_ee_force_estimator.train()

    def _setup_storage(self):
        super()._setup_storage()
        self.storage.register_key("left_ee_force_estimator_output", shape=(self.algo_obs_dim_dict["apply_left_ee_force"],), dtype=torch.float)
        self.storage.register_key("right_ee_force_estimator_output", shape=(self.algo_obs_dim_dict["apply_right_ee_force"],), dtype=torch.float)

    def _init_loss_dict_at_training_step(self):
        loss_dict = super()._init_loss_dict_at_training_step()
        loss_dict.update({
            "Left_EE_Force_Estimator_Loss": 0.0,
            "Right_EE_Force_Estimator_Loss": 0.0
        })
        return loss_dict

    def _update_algo_step(self, policy_state_dict, loss_dict):
        loss_dict = super()._update_algo_step(policy_state_dict, loss_dict)
        loss_dict = self._update_estimator(policy_state_dict, loss_dict)
        return loss_dict
    
    def _update_estimator(self, policy_state_dict, loss_dict):
        # map the force to [0, 1]
        left_ee_force = policy_state_dict["apply_left_ee_force"]
        right_ee_force = policy_state_dict["apply_right_ee_force"]
        # normalize the force for X, Y, Z directions
        left_ee_force_normalized = normalize(left_ee_force, self.force_min, self.force_max)
        right_ee_force_normalized = normalize(right_ee_force, self.force_min, self.force_max)
        
        # Update the estimator only when the force scale is greater than the threshold
        if self.config.update_estimator_only_when_stance:
            valid_idx = torch.where(
                (policy_state_dict["stance_command"] == 0) &
                (policy_state_dict["force_scale"] > self.config.update_estimator_force_scale_threshold)
            )[0]
        else:
            valid_idx = torch.where(
                (policy_state_dict["force_scale"] > self.config.update_estimator_force_scale_threshold))[0]

        if len(valid_idx) > 0:
            left_ee_force_estimator_target = left_ee_force_normalized[valid_idx]
            left_ee_force_estimator_output = self.left_ee_force_estimator(policy_state_dict["estimator_obs"])[valid_idx]
            right_ee_force_estimator_target = right_ee_force_normalized[valid_idx]
            right_ee_force_estimator_output = self.right_ee_force_estimator(policy_state_dict["estimator_obs"])[valid_idx]

            left_ee_force_estimator_loss = F.mse_loss(left_ee_force_estimator_output, left_ee_force_estimator_target)
            right_ee_force_estimator_loss = F.mse_loss(right_ee_force_estimator_output, right_ee_force_estimator_target)

            self.left_ee_force_estimator_optimizer.zero_grad()
            left_ee_force_estimator_loss.backward()
            nn.utils.clip_grad_norm_(self.left_ee_force_estimator.parameters(), self.max_grad_norm)
            self.left_ee_force_estimator_optimizer.step()

            self.right_ee_force_estimator_optimizer.zero_grad()
            right_ee_force_estimator_loss.backward()
            nn.utils.clip_grad_norm_(self.right_ee_force_estimator.parameters(), self.max_grad_norm)
            self.right_ee_force_estimator_optimizer.step()
        else:
            left_ee_force_estimator_loss = 0.0
            right_ee_force_estimator_loss = 0.0

        loss_dict['Left_EE_Force_Estimator_Loss'] += left_ee_force_estimator_loss
        loss_dict['Right_EE_Force_Estimator_Loss'] += right_ee_force_estimator_loss
        return loss_dict

    def _actor_rollout_step(self, obs_dict, policy_state_dict):
        actions, left_ee_force_estimator_output, right_ee_force_estimator_output = self._actor_act_step(obs_dict)
        policy_state_dict["actions"] = actions
        policy_state_dict["left_ee_force_estimator_output"] = left_ee_force_estimator_output
        policy_state_dict["right_ee_force_estimator_output"] = right_ee_force_estimator_output
        
        action_mean = self.actor.action_mean.detach()
        action_sigma = self.actor.action_std.detach()
        actions_log_prob = self.actor.get_actions_log_prob(actions).detach().unsqueeze(1)
        policy_state_dict["action_mean"] = action_mean
        policy_state_dict["action_sigma"] = action_sigma
        policy_state_dict["actions_log_prob"] = actions_log_prob

        assert len(actions.shape) == 2
        assert len(actions_log_prob.shape) == 2
        assert len(action_mean.shape) == 2
        assert len(action_sigma.shape) == 2

        return policy_state_dict

    def _actor_act_step(self, obs_dict):
        # Yuanhang: Use the estimated force as the actor input when the following conditions are met
        # 1. The apply force scale is greater than self.config.use_apply_or_estimated_force_scale_threshold 
        # 2. The stance command is 0 (0 for stance, 1 for walk)
        # Otherwise, use the apply force as the actor input because the estimated force is not reliable/accurate enough
        if self.config.update_estimator_only_when_stance:
            estimated_force_envs_id = torch.where(
                (obs_dict["force_scale"] > self.config.use_apply_or_estimated_force_scale_threshold) &
                (obs_dict["stance_command"] == 0)
            )[0]

            apply_force_envs_id = torch.where(
                (obs_dict["force_scale"] <= self.config.use_apply_or_estimated_force_scale_threshold) |
                (obs_dict["stance_command"] == 1)
            )[0]
        else:
            estimated_force_envs_id = torch.where(
                (obs_dict["force_scale"] > self.config.use_apply_or_estimated_force_scale_threshold)
            )[0]

            apply_force_envs_id = torch.where(
                (obs_dict["force_scale"] <= self.config.use_apply_or_estimated_force_scale_threshold)
            )[0]

        actor_obs_estimated = obs_dict["actor_obs"][estimated_force_envs_id]
        actor_obs_apply = obs_dict["actor_obs"][apply_force_envs_id]
        estimator_obs_estimated = obs_dict["estimator_obs"][estimated_force_envs_id]
        
        N = obs_dict['actor_obs'].shape[0]
        left_ee_force_estimator_output = normalize(torch.zeros((N, 3), device=self.device), self.force_min, self.force_max)
        right_ee_force_estimator_output = normalize(torch.zeros((N, 3), device=self.device), self.force_min, self.force_max)
        actor_input = torch.zeros((N,
                                   self.algo_obs_dim_dict["actor_obs"] + 6), device=self.device)
        if len(estimated_force_envs_id) > 0:
            left_ee_force_estimator_output[estimated_force_envs_id] = self.left_ee_force_estimator(estimator_obs_estimated).detach()
            right_ee_force_estimator_output[estimated_force_envs_id] = self.right_ee_force_estimator(estimator_obs_estimated).detach()
            actor_input[estimated_force_envs_id] = torch.cat([actor_obs_estimated,
                                                              left_ee_force_estimator_output[estimated_force_envs_id],
                                                              right_ee_force_estimator_output[estimated_force_envs_id]], dim=1)
        if len(apply_force_envs_id) > 0:
            apply_left_ee_force = obs_dict["apply_left_ee_force"][apply_force_envs_id]
            apply_right_ee_force = obs_dict["apply_right_ee_force"][apply_force_envs_id]
            
            # Normalize the apply force
            apply_left_ee_force_normalized = normalize(apply_left_ee_force, self.force_min, self.force_max)
            apply_right_ee_force_normalized = normalize(apply_right_ee_force, self.force_min, self.force_max)
            
            actor_input[apply_force_envs_id] = torch.cat([actor_obs_apply,
                                                          apply_left_ee_force_normalized,
                                                          apply_right_ee_force_normalized], dim=1)
        actions = self.actor.act(actor_input)
        # Update the use_apply_or_estimated_force
        self.use_apply_or_estimated_force = len(estimated_force_envs_id) / (len(estimated_force_envs_id) + len(apply_force_envs_id))
        if obs_dict.get("apply_or_estimated_force_flag") is not None:
            valid_estimated_force_envs_id = torch.where(
                (obs_dict["apply_or_estimated_force_flag"] == 1)
            )[0]
            self.valid_estimated_force_envs = len(valid_estimated_force_envs_id) / (len(estimated_force_envs_id) + len(apply_force_envs_id))

        return actions, left_ee_force_estimator_output, right_ee_force_estimator_output

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # Compute the actions and values
                # actions = self.actor.act(obs_dict["actor_obs"]).detach()

                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                ## Append states to storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])

                ## Get the lower body actions
                actions_lower_body = policy_state_dict["actions"]
                ## Get the upper body actions
                actions_upper_body = self.env.ref_upper_dof_pos
                ## Concatenate the lower and upper body actions
                actions = torch.cat([actions_lower_body, actions_upper_body], dim=1)
                actor_state = {"actions": actions, 
                               "left_ee_force_estimator_output": policy_state_dict["left_ee_force_estimator_output"], 
                               "right_ee_force_estimator_output": policy_state_dict["right_ee_force_estimator_output"]}
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                # critic_obs = privileged_obs if privileged_obs is not None else obs
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards.clone().unsqueeze(1)
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                assert len(rewards_stored.shape) == 2
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

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

    def _logging_to_writer(self, log_dict, train_log_dict, env_log_dict):
        super()._logging_to_writer(log_dict, train_log_dict, env_log_dict)
        # Log the action scale for the upper body
        self.writer.add_scalar('Env/use_apply_or_estimated_force', self.use_apply_or_estimated_force, log_dict['it'])
        if self.valid_estimated_force_envs:
            self.writer.add_scalar('Env/valid_estimated_force_envs', self.valid_estimated_force_envs, log_dict['it'])
    
    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################
    def _pre_eval_env_step(self, actor_state: dict):
        # left_ee_force_estimator_output = torch.zeros((actor_state["obs"]["actor_obs"].shape[0], 3), device=self.device)
        # right_ee_force_estimator_output = torch.zeros((actor_state["obs"]["actor_obs"].shape[0], 3), device=self.device)
        N = actor_state["obs"]['actor_obs'].shape[0]
        left_ee_force_estimator_output = normalize(torch.zeros((N, 3), device=self.device), self.force_min, self.force_max)
        right_ee_force_estimator_output = normalize(torch.zeros((N, 3), device=self.device), self.force_min, self.force_max)
        input_for_actor = torch.zeros((actor_state["obs"]["actor_obs"].shape[0],
                                       self.algo_obs_dim_dict["actor_obs"] + 6), device=self.device)
        # Yuanhang: Use the estimated force as the actor input ONLY when the stance command is 0
        # Otherwise, use the apply force as the actor input because the estimated force in walking is OOD
        if self.config.update_estimator_only_when_stance:
            stance_envs_idx = torch.where(actor_state["obs"]["stance_command"] == 0)[0]
            walk_envs_idx = torch.where(actor_state["obs"]["stance_command"] == 1)[0]
            left_ee_force_estimator_output[stance_envs_idx] = self.left_ee_force_estimator(actor_state["obs"]['estimator_obs'][stance_envs_idx])
            right_ee_force_estimator_output[stance_envs_idx] = self.right_ee_force_estimator(actor_state["obs"]['estimator_obs'][stance_envs_idx])
            actor_state.update({"left_ee_force_estimator_output": left_ee_force_estimator_output})
            actor_state.update({"right_ee_force_estimator_output": right_ee_force_estimator_output})
            input_for_actor[stance_envs_idx] = torch.cat([actor_state["obs"]['actor_obs'][stance_envs_idx], 
                                                          left_ee_force_estimator_output[stance_envs_idx], 
                                                          right_ee_force_estimator_output[stance_envs_idx]], dim=1)
            # Normalize the apply force
            apply_left_ee_force_normalized = normalize(actor_state["obs"]["apply_left_ee_force"], self.force_min, self.force_max)
            apply_right_ee_force_normalized = normalize(actor_state["obs"]["apply_right_ee_force"], self.force_min, self.force_max)
            input_for_actor[walk_envs_idx] = torch.cat([actor_state["obs"]['actor_obs'][walk_envs_idx],
                                                        apply_left_ee_force_normalized[walk_envs_idx],
                                                        apply_right_ee_force_normalized[walk_envs_idx]], dim=1)
        else:
            left_ee_force_estimator_output = self.left_ee_force_estimator(actor_state["obs"]['estimator_obs'])
            right_ee_force_estimator_output = self.right_ee_force_estimator(actor_state["obs"]['estimator_obs'])
            actor_state.update({"left_ee_force_estimator_output": left_ee_force_estimator_output})
            actor_state.update({"right_ee_force_estimator_output": right_ee_force_estimator_output})
            input_for_actor = torch.cat([actor_state["obs"]['actor_obs'], 
                                         left_ee_force_estimator_output, 
                                         right_ee_force_estimator_output], dim=1)
        actions = self.eval_policy(input_for_actor)
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state
    
    def env_step(self, actor_state):
        actions_lower_body = actor_state["actions"]
        actions_upper_body = self.env.ref_upper_dof_pos
        actions = torch.cat([actions_lower_body, actions_upper_body], dim=1)
        actor_state = {"step": actor_state["step"],
                       "actions": actions, 
                       "left_ee_force_estimator_output": actor_state["left_ee_force_estimator_output"], 
                       "right_ee_force_estimator_output": actor_state["right_ee_force_estimator_output"]}
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )
        return actor_state

    @property
    def inference_model(self):
        return {
            "actor": self.actor,
            "critic": self.critic,
            "left_ee_force_estimator": self.left_ee_force_estimator,
            "right_ee_force_estimator": self.right_ee_force_estimator
        }