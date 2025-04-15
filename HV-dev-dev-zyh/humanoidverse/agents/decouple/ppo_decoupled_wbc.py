import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.ppo.ppo import PPO
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

class PPODecoupledWBC(PPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        super().__init__(env, config, log_dir, device)
    
    def _logging_to_writer(self, log_dict, train_log_dict, env_log_dict):
        super()._logging_to_writer(log_dict, train_log_dict, env_log_dict)
        # Log the action scale for the upper body
        if hasattr(self.env, 'action_scale_upper_body'):
            self.writer.add_scalar('Env/action_scale_upper_body', torch.mean(self.env.action_scale_upper_body).item(), log_dict['it'])
        if hasattr(self.env, 'apply_force_scale'):
            self.writer.add_scalar('Env/apply_force_scale', torch.mean(self.env.apply_force_scale).item(), log_dict['it'])
        if hasattr(self.env, 'command_height_scale'):
            self.writer.add_scalar('Env/command_height_scale', torch.mean(self.env.command_height_scale).item(), log_dict['it'])