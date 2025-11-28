from collections import OrderedDict

import numpy as np
import tqdm

from gcb6206.critics.dqn_critic import DQNCritic
from gcb6206.critics.cql_critic import CQLCritic
from gcb6206.infrastructure.replay_buffer import ReplayBuffer
from gcb6206.infrastructure.utils import *
from gcb6206.infrastructure import pytorch_util as ptu
from gcb6206.policies.argmax_policy import ArgMaxPolicy
from gcb6206.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from .offline_agent import OfflineAgent


class CQLAgent(OfflineAgent):
    def __init__(self, env, agent_params):
        super(CQLAgent, self).__init__(env, agent_params)
        self.critic = CQLCritic(agent_params, self.optimizer_spec)
        self.actor = ArgMaxPolicy(self.critic)

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}

        env_reward = (re_n + self.rew_shift) * self.rew_scale

        # Update Critics #
        critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, env_reward, terminal_n)

        # Target Networks #
        if self.num_param_updates % self.target_update_freq == 0:
            self.critic.update_target_network()

        # Logging #
        log['Critic Loss'] = critic_loss['Training Loss']
        log['Data q-values'] = critic_loss['Data q-values']
        log['OOD q-values'] = critic_loss['OOD q-values']
        log['Overestimation'] = critic_loss['OOD q-values'] - critic_loss['Data q-values']
        log['CQL Loss'] = critic_loss['CQL Loss']

        self.num_param_updates += 1

        self.t += 1
        return log
