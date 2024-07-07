# q_learning.py
import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((env.observation_space.shape[0], env.action_space.n))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state]) * (1 - done)
        self.q_table[state, action] += self.alpha * (target - predict)

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, episodes):
        for e in range(episodes):
            state = self.env.reset()
            total_reward = 0

            for time in range(500):
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            print(f"Episode {e+1}/{episodes} - Reward: {total_reward}, Epsilon: {self.epsilon}")
