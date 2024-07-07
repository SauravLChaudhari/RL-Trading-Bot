# performance.py
import matplotlib.pyplot as plt
import numpy as np

class PerformanceEvaluator:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def evaluate(self, episodes):
        rewards = []
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.env.observation_space.shape[0]])
            total_reward = 0
            for time in range(500):
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.env.observation_space.shape[0]])
                total_reward += reward
                state = next_state
                if done:
                    break
            rewards.append(total_reward)
            print(f"Episode {e+1}/{episodes} - Reward: {total_reward}")
        return rewards

    def plot_rewards(self, rewards):
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Performance Evaluation')
        plt.show()
