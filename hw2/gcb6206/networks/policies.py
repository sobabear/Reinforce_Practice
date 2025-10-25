import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from gcb6206.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(parameters, learning_rate)

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # Convert numpy array to torch tensor
        obs = ptu.from_numpy(obs)
        
        # Get distribution from forward pass
        dist = self.forward(obs)
        
        # Sample action from distribution
        if self.discrete:
            # For discrete actions, return the integer directly
            action = dist.sample().item()
        else:
            action = dist.sample().cpu().numpy()
            
        return action

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # For discrete actions, use Categorical distribution
            logits = self.logits_net(obs)
            dist = distributions.Categorical(logits=logits)
        else:
            # For continuous actions, use Normal distribution
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            dist = distributions.Normal(mean, std)

        return dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        assert obs.ndim == 2
        assert advantages.ndim == 1
        assert obs.shape[0] == actions.shape[0] == advantages.shape[0]

        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # Clear gradients
        self.optimizer.zero_grad()
        
        # Get action distribution
        dist = self.forward(obs)
        
        # Calculate log probabilities
        log_probs = dist.log_prob(actions)
        
        # For discrete actions, log_prob returns shape (batch_size,)
        # For continuous actions with multiple dimensions, we need to sum across action dimensions
        if not self.discrete and log_probs.ndim > 1:
            log_probs = log_probs.sum(dim=-1)
        
        # Policy gradient loss: -log_prob * advantage
        loss = -(log_probs * advantages).mean()
        
        # Backpropagate and update
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }

    def ppo_update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_logp: np.ndarray,
        ppo_cliprange: float = 0.2,
    ) -> dict:
        """Implements the policy gradient actor update."""
        assert obs.ndim == 2
        assert advantages.ndim == 1
        assert old_logp.ndim == 1
        assert advantages.shape == old_logp.shape

        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        old_logp = ptu.from_numpy(old_logp)

        # TODO: Implement the ppo update.
        # HINT: calculate logp first, and then caculate ratio and clipped loss.
        # HINT: ratio is the exponential of the difference between logp and old_logp.
        # HINT: You can use torch.clamp to clip values.
        loss = None

        return {"PPO Loss": ptu.to_numpy(loss)}
