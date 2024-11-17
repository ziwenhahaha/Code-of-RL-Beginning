from gymnasium.spaces import Discrete
from typing_extensions import override
import numpy as np
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

desc = [
    "SFFFF",
    "FHHFF",
    "FFHFF",
    "FHGHF",
    "FHFFF",
]
env = gym.make(
    "FrozenLake-v1",
    desc=desc,
    map_name="5x5",
    is_slippery=False,
)
u_env = env.unwrapped


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def size(self):
        # 目前buffer中的元素个数
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)


class Qnet(torch.nn.Module):
    """只有一层隐藏层的Q网络"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    @override
    def forward(self, x):
        # print(f"{x.shape=}")
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate,
        gamma,
        epsilons,
        target_update,
        device,
    ) -> None:
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, 64, action_dim).to(device)
        self.target_net = Qnet(state_dim, 64, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilons = epsilons
        self.epsilon = epsilons[0]
        self.target_update = target_update  # target net 更新频率
        self.count = 0
        self.device = device
        self.loss = nn.MSELoss()

    def take_action(self, state) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.tensor([state], dtype=torch.float32).to(self.device)
                q_value = self.q_net(state)
                return int(torch.argmax(q_value).item())

    def update(self, transiton_dict):
        states = torch.tensor(
            transiton_dict["states"], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        actions = torch.tensor(
            transiton_dict["actions"], dtype=torch.long, device=self.device
        )[:, None]
        rewards = torch.tensor(
            transiton_dict["rewards"], dtype=torch.float32, device=self.device
        )[:, None]
        next_states = torch.tensor(
            transiton_dict["next_states"], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        dones = torch.tensor(
            transiton_dict["dones"], dtype=torch.float32, device=self.device
        )[:, None]

        q_values = self.q_net(states).gather(
            1, actions
        )  # 返回的是一个action_dim维的向量，取出对应的action value的值

        max_next_q_values = self.target_net(next_states).max(dim=1)[0][:, None]
        # 只使用一个网络,不使用target_net

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = self.loss(q_values, q_targets)
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    def decay_epsilon(self, i):
        self.epsilon = self.epsilons[i]

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)


lr = 1e-4
num_episodes = 40000
hidden_dim = 12
gamma = 0.99
target_update = 10
buffer_size = 10000
minimal_size = 1000
batch_size = 128
epsilon_decay = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
reply_buffer = ReplayBuffer(buffer_size)
state_dim = 1
action_dim = env.action_space.n

cutoff = 3000
epsilon = np.exp(-np.arange(num_episodes) / (cutoff))

epsilon[epsilon > epsilon[100 * int(num_episodes / cutoff)]] = epsilon[
    100 * int(num_episodes / cutoff)
]
agent = DQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)


max_v = -100
max_setp = 1000
for i in range(num_episodes):
    # break
    state = env.reset()[0]
    terminated = False
    truncated = False
    done = False
    now_reward = 0.0
    now_step = 0
    while not done and now_step < max_setp:
        now_step += 1
        action = agent.take_action(state)
        observation, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        print(f"{truncated=}, {terminated=}")
        if truncated:
            print(f"truncated episode {i}")
            reward = -1
        reply_buffer.add(state, action, reward, observation, done)
        state = observation
        now_reward += reward

        if len(reply_buffer) > minimal_size:
            b_s, b_a, b_r, b_ns, b_d = reply_buffer.sample(batch_size)
            transiton_dict = {
                "states": b_s,
                "actions": b_a,
                "rewards": b_r,
                "next_states": b_ns,
                "dones": b_d,
            }
            agent.update(transiton_dict)
    agent.decay_epsilon(i)
    # print(f"episode: {i}, reward: {now_reward}")
    if now_reward >= max_v:
        agent.save("DQN_model.pth")
        # print(f"update model episode {i} reward {now_reward}")
        max_v = now_reward
    if i % 100 == 0:
        print(f"episode: {i}, value: {now_reward}")

done = False
text_env = gym.make(
    "FrozenLake-v1",
    desc=desc,
    is_slippery=False,
    render_mode="human",
)
state = text_env.reset()[0]

best_net = Qnet(state_dim, 64, action_dim)
state = torch.tensor([state], dtype=torch.float32)
best_net.load_state_dict(torch.load("DQN_model.pth"))

while not done:
    q_value = best_net(state)
    action = int(torch.argmax(q_value).item())
    state, reward, terminated, truncated, info = text_env.step(action)
    state = torch.tensor([state], dtype=torch.float32)
    print(f"{state=},{action=}")
    text_env.render()  # 渲染环境
    done = terminated or truncated

text_env.close()
#
