import numpy as np
import gymnasium as gym


class ReturnWrapper(gym.Wrapper):
    def get_episode_rewards(self):
        return list(self.env.return_queue)
