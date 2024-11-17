import gymnasium as gym
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


class ValueIteration:
    def __init__(self, env, gamma, theta):
        self.env = env.unwrapped  # 获取未包装的原始环境
        self.gamma = gamma
        self.theta = theta
        self.n_states = env.observation_space.n  # 状态数量
        self.n_actions = env.action_space.n  # 动作数量
        self.v = np.zeros(self.n_states)  # 初始化价值函数
        self.pi = np.zeros(self.n_states, dtype=int)  # 初始化策略

    def value_iteration(self):
        while True:
            max_diff = 0
            new_v = np.copy(self.v)  # 创建新价值函数用于更新
            for state in range(self.n_states):
                qsa_list = []
                # 遍历所有可能的动作
                for action in range(self.n_actions):
                    qsa = 0
                    # 从当前状态采取action，计算Q值
                    for prob, next_state, reward, done in self.env.P[state][action]:
                        qsa += prob * (reward + self.gamma * self.v[next_state])
                    qsa_list.append(qsa)

                new_v[state] = max(qsa_list)  # 选择动作值最大的作为新价值
                max_diff = max(max_diff, abs(new_v[state] - self.v[state]))

            self.v = new_v
            if max_diff < self.theta:  # 如果最大变化小于阈值，停止迭代
                break

        self.get_policy()

    def get_policy(self):
        # 根据价值函数，提取最优策略
        for state in range(self.n_states):
            qsa_list = []
            for action in range(self.n_actions):
                qsa = 0
                for prob, next_state, reward, done in self.env.P[state][action]:
                    qsa += prob * (reward + self.gamma * self.v[next_state])
                qsa_list.append(qsa)
            self.pi[state] = np.argmax(qsa_list)  # 选择使Q值最大的动作

    def get_action(self, state):
        return self.pi[state]  # 返回最优动作


# 初始化价值迭代对象
vi = ValueIteration(env, gamma=gamma, theta=theta)

# 执行价值迭代
vi.value_iteration()

# 执行策略（交互）
observation, info = env.reset()
episode_over = False

while not episode_over:
    action = vi.get_action(observation)  # 根据最优策略选择动作
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()  # 渲染环境

    episode_over = terminated or truncated

env.close()
