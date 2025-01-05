#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :agent.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import os
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
)
from diy.model.model import ActorNet, CriticNet
from diy.feature.definition import ActData
from kaiwu_agent.utils.common_func import attached
from diy.config import Config
import numpy as np

@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        self.talent_direction = Config.DIM_OF_TALENT
        self.obs_shape = Config.DIM_OF_OBSERVATION
        # self.epsilon = Config.EPSILON
        # self.egp = Config.EPSILON_GREEDY_PROBABILITY
        self.obs_split = Config.DESC_OBS_SPLIT
        self._gamma = Config.GAMMA
        self.actor_lr = Config.ACTOR_START_LR
        self.critic_lr = Config.CRITIC_START_LR

        self.current_end = None


        self.device = device
        self.actor = ActorNet(
            state_shape=self.obs_shape,
            action_shape=self.act_shape,
            softmax=False,
        )
        self.critic = CriticNet(
            state_shape=self.obs_shape,
            action_shape=1,
            softmax=False,
        )

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0

        self.agent_type = agent_type
        self.logger = logger
        self.monitor = monitor
        self.action_prob = None

        if Config.DEBUG == True:
            directory = './diy/ckpt/'

            # 初始化一个空列表，用于存储文件名中的数字
            model_ids = []

            # 遍历目录中的所有文件
            for filename in os.listdir(directory):
                # 使用正则表达式提取文件名中的数字
                match = re.search(r'\d+', filename)
                if match:
                    model_ids.append(int(match.group()))
            latest_model_id = max(model_ids)
            self.__load_model(path=directory, id=latest_model_id)
            logger.info(f"now loading id == {latest_model_id}")
            logger.info(f"id {latest_model_id} load succeed")

    def modify_end(self, current_end):
        self.current_end = current_end
        data_dict = {
            1: {"norm_x": 0.8203125, "grid_x": 105, "norm_z": 0.7421875, "grid_z": 109},
            2: {"norm_x": 0.0859375, "grid_x": 13, "norm_z": 0.0234375, "grid_z": 15},
            3: {"norm_x": 0.5703125, "grid_x": 11, "norm_z": 0.0078125, "grid_z": 77},
            4: {"norm_x": 0.7734375, "grid_x": 55, "norm_z": 0.3515625, "grid_z": 103},
            5: {"norm_x": 0.2421875, "grid_x": 113, "norm_z": 0.8046875, "grid_z": 35},
            6: {"norm_x": 0.5390625, "grid_x": 45, "norm_z": 0.2734375, "grid_z": 73},
            7: {"norm_x": 0.2890625, "grid_x": 19, "norm_z": 0.0703125, "grid_z": 41},
            8: {"norm_x": 0.4296875, "grid_x": 117, "norm_z": 0.8359375, "grid_z": 59},
            9: {"norm_x": 0.4296875, "grid_x": 81, "norm_z": 0.5546875, "grid_z": 59},
            10: {"norm_x": 0.2265625, "grid_x": 53, "norm_z": 0.3359375, "grid_z": 33},
            11: {"norm_x": 0.7578125, "grid_x": 31, "norm_z": 0.1640625, "grid_z": 101},
            12: {"norm_x": 0.0859375, "grid_x": 97, "norm_z": 0.6796875, "grid_z": 15},
            13: {"norm_x": 0.8046875, "grid_x": 83, "norm_z": 0.5703125, "grid_z": 107},
            14: {"norm_x": 0.1328125, "grid_x": 73, "norm_z": 0.4921875, "grid_z": 21},
            15: {"norm_x": 0.0703125, "grid_x": 41, "norm_z": 0.2421875, "grid_z": 13},
        }
        data_dict[current_end]["norm_x"]
        data_dict[current_end]["norm_z"]
        data_dict[current_end]["grid_x"]
        data_dict[current_end]["grid_z"]
        
    def __convert_to_tensor(self, data):
        if isinstance(data, list):
            return torch.tensor(
                np.array(data),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            return torch.tensor(
                data,
                device=self.device,
                dtype=torch.float32,
            )
        
    def __legal_soft_max(self, logits, legal_acts):
        assert logits.shape == legal_acts.shape
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5

        tmp = logits - _lsm_const_w * (1.0 - legal_acts.float())
        tmp_max = torch.max(tmp, dim=-1, keepdims=True)[0]
        tmp = torch.clamp(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (torch.exp(tmp) + _lsm_const_e) * legal_acts
        probs = tmp / torch.sum(tmp, dim=-1, keepdims=True)
        return probs

    @predict_wrapper
    def predict(self, list_obs_data):
        if list_obs_data[0].feature[0]==list_obs_data[0].feature[260] and list_obs_data[0].feature[1]==list_obs_data[0].feature[261]:
            #若小概率事件predict了，那么用agent的数据补上
            pass

        batch = len(list_obs_data)
        feature_vec = [obs_data.feature[: self.obs_split[0]] for obs_data in list_obs_data]
        feature_map = [obs_data.feature[self.obs_split[0] :] for obs_data in list_obs_data]
        legal_act = [obs_data.legal_act for obs_data in list_obs_data]
        legal_act = torch.tensor(np.array(legal_act))
        legal_act = (
            torch.cat(
                (
                    legal_act[:, 0].unsqueeze(1).expand(batch, self.direction_space),
                    legal_act[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
                ),
                1,
            )
            .bool()
            .to(self.device)
        )

        actor = self.actor
        actor.eval()
        action_prob = 0.001

        with torch.no_grad():
            feature = [
                self.__convert_to_tensor(feature_vec),
                self.__convert_to_tensor(feature_map).view(batch, *self.obs_split[1]),
            ]
            logits, _ = actor(feature, state=None)
            legal_probs = self.__legal_soft_max(logits, legal_act)

            c = Categorical(legal_probs)
            act = c.sample()
            action_prob = legal_probs.gather(1, act.view(-1,1)).item()
            act = act.cpu().view(-1, 1).tolist()

        # 这里是为了区分闪现还是走路，由第1维度的值（从第0维度开始）来区分
        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]

        self.predict_count += 1

        return [ActData(move_dir=i[0], use_talent=i[1], prob=action_prob) for i in format_action]




    @exploit_wrapper
    def exploit(self, list_obs_data):
        return None


    @learn_wrapper
    def learn(self, list_sample_data):
        t_data = list_sample_data
        batch = len(t_data)

        batch_feature_vec = [frame.obs[: self.obs_split[0]] for frame in t_data]
        batch_feature_map = [frame.obs[self.obs_split[0] :] for frame in t_data]
        batch_action = torch.LongTensor(np.array([int(frame.act) for frame in t_data])).view(-1, 1).to(self.device)
        batch_prob = torch.tensor(np.array([float(frame.prob) for frame in t_data]).astype(np.float32)).view(-1, 1).to(self.device)


        #############debug
        probs_np = batch_prob.detach().cpu().numpy()
        
        # Find indices where probabilities are very close to 0 or 1
        close_to_zero = (probs_np < 1e-5)
        close_to_one = (probs_np > 1 - 1e-5)
        
        # Print some examples of probabilities close to 0
        if np.any(close_to_zero):
            self.logger.info(f"Probabilities close to 0: {probs_np[close_to_zero][:10].tolist()}")
        
        # Print some examples of probabilities close to 1
        if np.any(close_to_one):
            self.logger.info(f"Probabilities close to 1: {probs_np[close_to_one][:10].tolist()}")
        
        # Optional: Find samples with high standard deviation
        std_devs = np.std(probs_np, axis=-1)
        high_std_indices = np.argsort(-std_devs)[:5]  # Get indices of top 5 highest std devs
        self.logger.info(f"Samples with high standard deviation: {probs_np[high_std_indices].tolist()}")
        
        #############debug
        batch_obs_legal = torch.tensor(np.array([frame.obs_legal for frame in t_data]))
        batch_obs_legal = (
            torch.cat(
                (
                    batch_obs_legal[:, 0].unsqueeze(1).expand(batch, self.direction_space),
                    batch_obs_legal[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
                ),
                1,
            )
            .bool()
            .to(self.device)
        )
        ret = torch.tensor(np.array([frame.ret for frame in t_data]), device=self.device)

        batch_feature = [
            self.__convert_to_tensor(batch_feature_vec),
            self.__convert_to_tensor(batch_feature_map).view(batch, *self.obs_split[1]),
        ]

        actor = getattr(self, "actor")
        critic = getattr(self, "critic")
        actor.eval()
        critic.eval()

        logits, h = actor(batch_feature, state=None)
        # masked_logits = logits.masked_fill(~batch_obs_legal, float("-inf"))
        # all_action_prob_batch = F.softmax(masked_logits, dim=1)
        all_action_prob_batch = self.__legal_soft_max(logits, batch_obs_legal)

        s_batch = batch_feature
        a_batch = batch_action
        old_action_prob_batch = batch_prob
        G_batch = ret.view(-1, 1)
        V_batch = critic(s_batch)[0]
        delta = G_batch - V_batch
        advantage = delta.detach().clone()

        entropy_ = Categorical(all_action_prob_batch).entropy().mean()
        action_prob_batch = all_action_prob_batch.gather(1, a_batch) 
        ratio = (action_prob_batch / old_action_prob_batch)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - Config.CLIP_PARAM, 1 + Config.CLIP_PARAM) * advantage
        action_loss = -torch.min(surr1, surr2).mean()
        act_loss = action_loss - Config.ENTROPY_COEF * entropy_

        if torch.isnan(act_loss).any():
            return

        actor.train()
        critic.train()
        self.actor_optimizer.zero_grad()
        act_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), Config.MAX_GRAD_NORM) 
        self.actor_optimizer.step()
        value_loss = F.mse_loss(V_batch, G_batch)
        self.critic_optimizer.zero_grad()
        value_loss.backward()  
        nn.utils.clip_grad_norm_(self.critic.parameters(), Config.MAX_GRAD_NORM)
        self.critic_optimizer.step()

        self.train_step += 1

        actor_loss = action_loss.item()
        critic_loss = value_loss.item()
        entropy = entropy_.item()
        ret = ret.mean().detach().item()

        # self.logger.info(f"probs: {probs.tolist()}")
        self.logger.info(f"actor_loss:{actor_loss} \n critic_loss:{critic_loss} \n entropy:{entropy} \n ret:{ret}")

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                # "value_loss": actor_loss,
                # "q_value": critic_loss,
                # "reward": entropy,
                "diy_1": actor_loss,
                "diy_2": critic_loss,
                "diy_3": entropy,
                "diy_4": ret,
                "diy_5": 0,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        actor_model_file_path = f"{path}/ActorNet_model.ckpt-{str(id)}.pkl"
        actor_model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.actor.state_dict().items()}
        torch.save(actor_model_state_dict_cpu, actor_model_file_path)
        critic_model_file_path = f"{path}/CriticNet_model.ckpt-{str(id)}.pkl"
        critic_model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.critic.state_dict().items()}
        torch.save(critic_model_state_dict_cpu, critic_model_file_path)

        self.logger.info(f"save ActorNet {actor_model_file_path} successfully")
        self.logger.info(f"save CriticNet {critic_model_file_path} successfully")
        pass

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        actor_model_file_path = f"{path}/ActorNet_model.ckpt-{str(id)}.pkl"
        self.logger.info(f"actor_model_file_path:{actor_model_file_path}")
        self.actor.load_state_dict(torch.load(actor_model_file_path, map_location=self.device))
        critic_model_file_path = f"{path}/CriticNet_model.ckpt-{str(id)}.pkl"
        self.critic.load_state_dict(torch.load(critic_model_file_path, map_location=self.device))

    def __load_model(self, path=None, id="1"):
        actor_model_file_path = f"{path}/ActorNet_model.ckpt-{str(id)}.pkl"
        self.logger.info(f"actor_model_file_path:{actor_model_file_path}")
        self.actor.load_state_dict(torch.load(actor_model_file_path, map_location=self.device))
        critic_model_file_path = f"{path}/CriticNet_model.ckpt-{str(id)}.pkl"
        self.critic.load_state_dict(torch.load(critic_model_file_path, map_location=self.device))
        self.logger.info(f"load ActorNet {actor_model_file_path} successfully")
        self.logger.info(f"load CriticNet {critic_model_file_path} successfully")