import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        if len(state_dim) == 1:
            self.is_cnn = False
            self.net1 = nn.Sequential(
                nn.Linear(state_dim[0], hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
            )
            self.net2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
        else:
            self.is_cnn = True
            self.net1 = nn.Sequential(
                nn.Conv2d(state_dim[-1], 25, 5, 1, "same"),
                nn.LeakyReLU(),
                nn.Conv2d(25, 25, 3, 1, "same"),
                nn.LeakyReLU(),
                nn.Conv2d(25, 25, 3, 1, "valid"),
                nn.LeakyReLU(),
                nn.Flatten(),
            )
            self.net2 = nn.Sequential(
                nn.Linear(25 * 3 * 2, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
    
    def forward(self, x):
        if self.is_cnn:
            x = x.permute(0,3,1,2)
        x = self.net1(x)
        return F.softmax(self.net2(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        if len(state_dim) == 1:
            self.is_cnn = False
            self.net1 = nn.Sequential(
                nn.Linear(state_dim[0], hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
            )
            self.net2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.is_cnn = True
            self.net1 = nn.Sequential(
                nn.Conv2d(state_dim[-1], 25, 5, 1, "same"),
                nn.LeakyReLU(),
                nn.Conv2d(25, 25, 3, 1, "same"),
                nn.LeakyReLU(),
                nn.Conv2d(25, 25, 3, 1, "valid"),
                nn.LeakyReLU(),
                nn.Flatten(),
            )
            self.net2 = nn.Sequential(
                nn.Linear(25 * 3 * 2, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1),
            )
    
    def forward(self, x):
        if self.is_cnn:
            x = x.permute(0,3,1,2)
        x = self.net1(x)
        return self.net2(x)


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, device,
                # the keyword arguments is only needed for pretraining / finetuning
                    num_steps=100, batch_size=4096, actor_lr=1e-4, critic_lr=1e-4, entropy_coef=1e-4, gamma=0.99, 
                    num_update_per_iter=10, clip_param=0.2, max_grad_norm=5.0, buffer_name=None,
                    ):
        super(PPO, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.actor_net = PolicyNet(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.critic_net = ValueNet(self.state_dim, self.hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_update_per_iter = num_update_per_iter
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.buffer = {name:[] for name in buffer_name} if buffer_name is not None else []
        self.buffer_size = 0
        self.training_step = 0
        
    def select_action(self, state):
        if state.dtype == np.object_:
            state = np.array(state.tolist(), dtype=np.float32)
        # bs, obs_dim
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            # bs, act_dim
            all_action_prob = self.actor_net(state)
        c = Categorical(all_action_prob)
        # bs
        action = c.sample()
        # bs, act_dim
        action_onehot = F.one_hot(action, self.action_dim).float()
        # bs
        action_prob = all_action_prob.gather(1, action.view(-1,1)).squeeze(1)
        return action_onehot.cpu().numpy(), action.cpu().numpy(), action_prob.cpu().numpy(), all_action_prob.cpu().numpy()
    
    def save_params(self, path):
        save_dict = {'actor': self.actor_net.state_dict(), 'critic': self.critic_net.state_dict()}
        name = path+'.pt'
        torch.save(save_dict, name, _use_new_zipfile_serialization=False)

    def load_params(self, filename):
        save_dict = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename), map_location=self.device)
        self.actor_net.load_state_dict(save_dict['actor'])
        self.critic_net.load_state_dict(save_dict['critic'])
    
    def load_params_from_policy(self, policy):
        self.actor_net.load_state_dict(policy.actor_net.state_dict())
        self.critic_net.load_state_dict(policy.critic_net.state_dict())

    def store_transition(self, transition, j):
        self.buffer[j].append(transition)
        self.buffer_size += len(transition.state)
    
    def update(self):
        state, action, old_action_prob, G = [], [], [], []
        for name, buffer in self.buffer.items():
            # state_: T, num_train_ep, obs_dim
            # action_: T, num_train_ep, 1
            # old_action_prob_: T, num_train_ep, 1
            # reward_: T, num_train_ep
            state_ = torch.tensor(np.array([t.state for t in buffer]), dtype=torch.float).to(self.device)
            action_ = torch.tensor(np.array([t.action for t in buffer]), dtype=torch.long).to(self.device)
            old_action_prob_ = torch.tensor(np.array([t.a_prob for t in buffer]), dtype=torch.float).to(self.device)
            reward_ = [t.reward for t in buffer]
            
            if len(state_) == 0:
                continue
            
            num_train_ep = state_.shape[1]

            R = np.zeros((num_train_ep))
            G_ = []
            count = 0
            for r in reward_[::-1]:
                R = r + self.gamma * R
                count += 1
                G_.insert(0, R)
                if count >= self.num_steps:
                    R = np.zeros((num_train_ep))
                    count = 0
            # G_: T, num_train_ep
            G_ = torch.tensor(np.array(G_), dtype=torch.float).to(self.device)
            
            state.append(state_.view(-1, *state_.shape[2:]))
            action.append(action_.view(-1,1))
            old_action_prob.append(old_action_prob_.view(-1,1))
            G.append(G_.view(-1))
        state, action, old_action_prob, G = torch.cat(state), torch.cat(action), torch.cat(old_action_prob), torch.cat(G)
        
        actor_loss, critic_loss, entropy, loss_count = 0., 0., 0., 0
        for _ in range(self.num_update_per_iter):
            for index in BatchSampler(SubsetRandomSampler(range(len(state))), self.batch_size, False):
                s_batch = state[index]
                a_batch = action[index]
                old_action_prob_batch = old_action_prob[index]
                G_batch = G[index].view(-1,1)
                V_batch = self.critic_net(s_batch)
                delta = G_batch - V_batch
                advantage = delta.detach().clone()

                all_action_prob_batch = self.actor_net(s_batch)
                entropy_ = Categorical(all_action_prob_batch).entropy().mean()
                action_prob_batch = all_action_prob_batch.gather(1, a_batch) # new policy
                ratio = (action_prob_batch / old_action_prob_batch)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean() # Max->Min desent
                act_loss = action_loss - self.entropy_coef * entropy_
                self.actor_optimizer.zero_grad()
                act_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(V_batch, G_batch)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                self.training_step += 1
                
                actor_loss += action_loss.item()
                critic_loss += value_loss.item()
                entropy += entropy_.item()
                
                loss_count += 1

        # clear experience
        for name in self.buffer:
            self.buffer[name] = []
        self.buffer_size = 0
        
        torch.cuda.empty_cache()
        
        return actor_loss/loss_count, critic_loss/loss_count, entropy/loss_count


class RandomAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, device,
                 # 这些参数只是为了与PPO保持一致
                 num_steps=100, batch_size=4096, actor_lr=1e-4, critic_lr=1e-4, entropy_coef=1e-4, gamma=0.99, 
                 num_update_per_iter=10, clip_param=0.2, max_grad_norm=5.0, buffer_name=None):
        self.device = device
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
    def select_action(self, state):
        if state.dtype == np.object_:
            state = np.array(state.tolist(), dtype=np.float32)
        # bs, obs_dim
        state = torch.from_numpy(state).float().to(self.device)
        batch_size = state.shape[0]
        
        # 随机生成动作
        action = torch.randint(0, self.action_dim, (batch_size,)).to(self.device)
        action_onehot = F.one_hot(action, self.action_dim).float()
        
        # 动作的均匀概率
        all_action_prob = torch.full((batch_size, self.action_dim), 1.0 / self.action_dim).to(self.device)
        action_prob = all_action_prob.gather(1, action.view(-1, 1)).squeeze(1)
        
        return action_onehot.cpu().numpy(), action.cpu().numpy(), action_prob.cpu().numpy(), all_action_prob.cpu().numpy()