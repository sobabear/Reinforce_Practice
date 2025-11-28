from collections import OrderedDict

import tqdm
import numpy as np
import torch

from gcb6206.critics.dqn_critic import DQNCritic
from gcb6206.infrastructure.replay_buffer import ReplayBuffer
from gcb6206.infrastructure.utils import *
from gcb6206.infrastructure import pytorch_util as ptu
from gcb6206.policies.argmax_policy import ArgMaxPolicy
from gcb6206.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from .dqn_agent import DQNAgent


class OfflineAgent(DQNAgent):

    def __init__(self, env, agent_params):
        super(OfflineAgent, self).__init__(env, agent_params)

        self.dataset_size = agent_params['dataset_size']
        self.replay_buffer = ReplayBuffer(100000)

        self.rew_shift = agent_params['rew_shift']
        self.rew_scale = agent_params['rew_scale']
        self.eps = agent_params['eps']
        if agent_params['dataset_name'] == 'replay':
            self.gather_replay_data(agent_params)
        else:
            self.gather_data(agent_params['dataset_name'])

    def gather_data(self, dataset_name):
        paths = []
        if 'random' == dataset_name:
            def random_actor(obs, step):
                action = self.env.action_space.sample()
                return action
            paths.extend(self.get_paths(random_actor, 100000)) # self.dataset_size))

        if 'expert' == dataset_name:
            def expert_actor(obs, step):
                if np.random.random() < 0.0: #0.3: #self.eps:
                    action = self.env.action_space.sample()
                else:
                    action = self.env.get_optimal_action(obs)
                return action
            paths.extend(self.get_paths(expert_actor, 20000)) #self.dataset_size))
        self.replay_buffer.add_rollouts(paths)

    def get_paths(self, actor, dataset_size):
        paths, path = [], []
        obs, _ = self.env.reset()
        for step in tqdm.tqdm(range(dataset_size)):
            action = actor(obs, step)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            path.append({
                'observations': obs,
                'actions': action,
                'rewards': reward,
                'terminals': terminated,
                'next_observations': next_obs
            })
            if terminated or truncated:
                obs, _ = self.env.reset()
                paths.append({k: np.stack([path[i][k] for i in range(len(path))],axis=0) for k in path[0]})
                path = []
            else:
                obs = next_obs
        return paths

