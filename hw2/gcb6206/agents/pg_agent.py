from typing import Optional, Sequence
import numpy as np
import torch

from gcb6206.networks.policies import MLPPolicyPG
from gcb6206.networks.critics import ValueCritic
from gcb6206.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
        use_ppo: bool = False,
        n_ppo_epochs: int = 4,
        n_ppo_minibatches: int = 4,
        ppo_cliprange: float = 0.2,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

        self.use_ppo = use_ppo
        self.n_ppo_epochs = n_ppo_epochs if self.use_ppo else None
        self.n_ppo_minibatches = n_ppo_minibatches if use_ppo else None
        self.ppo_cliprange = ppo_cliprange if use_ppo else None

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        assert all(ob.ndim == 2 for ob in obs)
        assert all(reward.ndim == 1 for reward in rewards)
        assert all(terminal.ndim == 1 for terminal in terminals)

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # Flatten lists of arrays into single arrays
        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        # q_values is already a numpy array from _calculate_q_vals, so no need to convert

        # step 2: calculate advantages from Q values
        assert q_values.ndim == 1
        advantages = self._estimate_advantage(obs, rewards, q_values, terminals)

        assert advantages.ndim == 1
        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        if not self.use_ppo:
            # Normalize the advantages to have mean 0 and std 1
            if self.normalize_advantages:
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

            # Update the PG actor/policy network once using the advantages
            info = self.actor.update(obs, actions, advantages)

            if self.critic is not None:
                # TODO: update the critic for `baseline_gradient_steps` times
                critic_info = {}
                for _ in range(self.baseline_gradient_steps):
                    critic_update_info = self.critic.update(obs, q_values)
                    # Average the losses over all gradient steps
                    for key, value in critic_update_info.items():
                        if key not in critic_info:
                            critic_info[key] = value / self.baseline_gradient_steps
                        else:
                            critic_info[key] += value / self.baseline_gradient_steps

                info.update(critic_info)
        else:
            # skip this part until you implement PPO
            # for ppo updates, we need to calculate the log probabilities of old policies
            logp: np.ndarray = self._calculate_log_probs(obs, actions)
            assert logp.ndim == 1

            # gradient steps on minibatches
            n_batch = len(obs)
            inds = np.arange(n_batch)
            for _ in range(self.n_ppo_epochs):
                np.random.shuffle(inds)
                # calculate minibatch size to divide a batch to `n_ppo_minibatches` minibatches
                minibatch_size = (
                    n_batch + (self.n_ppo_minibatches - 1)
                ) // self.n_ppo_minibatches
                for start in range(0, n_batch, minibatch_size):
                    end = start + minibatch_size
                    obs_slice, actions_slice, advantages_slice, logp_slice = (
                        arr[inds[start:end]] for arr in (obs, actions, advantages, logp)
                    )

                    # TODO: normalize `advantages_slice`` to have a mean of zero and a standard deviation of one within the batch
                    if self.normalize_advantages:
                        pass

                    # TODO: update the PG actor/policy with PPO objective
                    # HINT: call self.actor.ppo_update
                    info: dict = None

            assert self.critic is not None, "PPO requires a critic for calculating GAE."
            # TODO: update the critic for `baseline_gradient_steps` times
            critic_info: dict = None

            info.update(critic_info)
        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        assert all(reward.ndim == 1 for reward in rewards)

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # Calculate Q-values using discounted returns for each trajectory and concatenate
            q_values = np.concatenate([self._discounted_return(reward) for reward in rewards])
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # Calculate Q-values using reward-to-go for each trajectory and concatenate
            q_values = np.concatenate([self._discounted_reward_to_go(reward) for reward in rewards])

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values."""

        assert obs.ndim == 2

        if self.critic is None:
            # Without baseline, advantages are just the Q-values
            advantages = q_values
        else:
            # TODO: run the critic and use it as a baseline
            values = self.critic(obs).squeeze()

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                # HINT: calculating `deltas` as in the GAE formula first would be useful.
                # HINT2: handle edge cases by using `terminals`. You can multiply (1 - terminals) to the value of the next state
                # to handle this.

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if there isn't a next state in its
                    # trajectory, and 0 otherwise.
                    pass

                # remove dummy advantage
                advantages = advantages[:-1]

        return advantages

    def _discounted_return(self, rewards: np.ndarray[float]) -> np.ndarray[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!

        Example:
        ```python
        # assume gamma = 0.99
        rewards = np.array([1., 2., 3.])
        total_discounted_return = agent._discounted_return(rewards)
        print(total_discounted_return)
        ```

        Output:
        ```
        np.array([5.9203, 5.9203, 5.9203])
        ```
        """
        assert rewards.ndim == 1
        # Calculate discounted return for each timestep
        T = len(rewards)
        ret = np.zeros_like(rewards)
        
        # For each timestep, sum up all future rewards with discount
        discounted_sum = 0
        for t in range(T):
            discounted_sum += (self.gamma ** t) * rewards[t]
        
        # All timesteps get the same value (sum from 0 to T)
        ret.fill(discounted_sum)
        
        assert rewards.shape == ret.shape
        return ret

    def _discounted_reward_to_go(self, rewards: np.ndarray[float]) -> np.ndarray[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.

        Example:
        ```python
        # assume gamma = 0.99
        rewards = np.array([1., 2., 3.])
        total_discounted_return = agent._discounted_reward_to_go(rewards)
        print(total_discounted_return)
        ```

        Output:
        ```
        np.array([5.9203, 4.97, 3.])
        ```
        """
        assert rewards.ndim == 1
        # Calculate reward to go for each timestep
        T = len(rewards)
        ret = np.zeros_like(rewards)
        
        # Start from the back and accumulate rewards
        running_sum = 0
        for t in reversed(range(T)):
            running_sum = rewards[t] + self.gamma * running_sum
            ret[t] = running_sum
            
        assert rewards.shape == ret.shape
        return ret

    def _calculate_log_probs(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
    ):
        """
        Helper function to calculate the log probabilities of the actions taken by the actor.
        """

        assert obs.ndim == 2
        # TODO: calculate the log probabilities
        # HINT: self.actor outputs a distribution object, which has a method log_prob that takes in the actions
        logp = None

        assert logp.ndim == 1 and logp.shape[0] == obs.shape[0]
        return logp
