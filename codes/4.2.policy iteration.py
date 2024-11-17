import gymnasium as gym
from gymnasium.spaces import Discrete
import copy
import numpy as np

# 定义FrozenLake环境的地图
desc = [
    "SFFFF",
    "FHHFF",
    "FFHFF",
    "FHGHF",
    "FHFFF",
]
env = gym.make(
    "FrozenLake-v1", desc=desc, map_name="5x5", is_slippery=False, render_mode="human"
)

gamma = 0.9  # 折扣因子
rows, columns = 5, 5  # 行列数
theta = 1e-6  # 阈值，用于停止价值迭代


class PolicyIteration:
    def __init__(self, env: gym.Env[Discrete,Discrete], theta: float, gamma: float):
        self.env:gym.Env[Discrete, Discrete] = env.unwrapped
        self.theta = theta
        self.gamma = gamma
        self.n_states = env.observation_space.n  # 状态数量
        self.n_actions = env.action_space.n  # 动作数量
        self.v = np.zeros(self.n_states)  # 初始化价值函数
        self.pi = np.zeros(self.n_states, dtype=int)  # 初始化策略

    def policy_evaluation(self):
        while True:
            new_v = np.copy(self.v)
            max_diff = 0
            cnt = 1
            for state in range(self.n_states):
                action = self.pi[state]
                _, next_state, reward, _ = self.env.P[state][action][0]

                action_value = reward + self.gamma * self.v[next_state]
                new_v[state] = action_value

                max_diff = max(max_diff, abs(new_v[state] - self.v[state]))
            cnt += 1
            print(new_v)
            self.v = new_v
            if max_diff < self.theta:
                print("policy evaluation iteration times = :", cnt)
                break

    def policy_improvement(self):
        for state in range(self.n_states):
            q = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                _, next_state, reward, _ = self.env.P[state][a][0]
                q[a] = reward + self.gamma * self.v[next_state]
                new_a = np.argmax(q)
                self.pi[state] = new_a
        return self.pi

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if np.all(old_pi == new_pi):
                break


pi = PolicyIteration(env, gamma=gamma, theta=theta)


# 执行价值迭代
pi.policy_iteration()

# 执行策略（交互）
observation, info = env.reset()
episode_over = False
print(pi.pi)
while not episode_over:
    action = pi.pi[observation]  # 根据最优策略选择动作
    print(type(action))
    # print(action, observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()  # 渲染环境

    episode_over = terminated or truncated

env.close()
#
