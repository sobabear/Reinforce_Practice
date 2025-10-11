"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    1. get_action (line 111)
    2. forward (line 126)
    3. update (line 141)
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from gcb6206.infrastructure import pytorch_util as ptu
from gcb6206.policies.base_policy import BasePolicy


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to actions

    Attributes
    ----------
    logits_na: nn.Sequential
        A neural network that outputs dicrete actions
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    get_action:
        Calls the actor forward function
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """

    def __init__(
        self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        discrete=False,
        learning_rate=1e-4,
        training=True,
        nn_baseline=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(), self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate,
            )

    ##################################

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        :param obs: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # Convert numpy observation to torch tensor
        observation = ptu.from_numpy(observation)
        
        # Get action distribution from policy network
        action_distribution = self.forward(observation)
        
        # Sample action from the distribution
        action = action_distribution.sample()
        
        # Convert action to numpy array
        return ptu.to_numpy(action)

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # Get mean from the mean network
        mean = self.mean_net(observation)
        
        # Convert logstd to std by exponentiating
        std = torch.exp(self.logstd)
        
        # Create and return a Normal distribution
        return distributions.Normal(mean, std)

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # Convert numpy arrays to torch tensors
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Get action distribution
        action_distribution = self.forward(observations)
        
        # Calculate negative log likelihood loss
        loss = -action_distribution.log_prob(actions).mean()
        
        # Backpropagate and update
        loss.backward()
        self.optimizer.step()
        
        return {
            "Training Loss": ptu.to_numpy(loss),
        }
