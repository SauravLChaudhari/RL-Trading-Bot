# trading_env.py
import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.n_steps = len(data)
        self.current_step = 0

        # Define action and observation space
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        # Observations: price, moving average, etc.
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        obs = np.array([
            self.data['Close'].iloc[self.current_step],
            self.data['MA'].iloc[self.current_step],
            self.data['Volume'].iloc[self.current_step]
        ])
        return obs

    def step(self, action):
        self.current_step += 1
        if self.current_step >= self.n_steps:
            self.current_step = 0

        reward = self._take_action(action)
        done = self.current_step == self.n_steps - 1

        obs = self._next_observation()
        return obs, reward, done, {}

    def _take_action(self, action):
        # Define reward function
        reward = 0
        if action == 1:  # Buy
            reward = self.data['Close'].iloc[self.current_step] - self.data['Close'].iloc[self.current_step - 1]
        elif action == 2:  # Sell
            reward = self.data['Close'].iloc[self.current_step - 1] - self.data['Close'].iloc[self.current_step]
        return reward
