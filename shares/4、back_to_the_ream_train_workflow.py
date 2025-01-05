#!/usr/bin/env python3
# -*- coding:utf-8 -*-


"""
@Project :back_to_the_realm
@File    :train_work_flow.py
@Author  :kaiwu
@Date    :2022/11/15 20:57

"""

import time
import os
import math
import re
import numpy as np
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
from diy.feature.definition import (
    observation_process,
    action_process,
    sample_process,
    reward_shaping,
)
from kaiwu_agent.back_to_the_realm.dqn.feature_process import (
    one_hot_encoding,
    read_relative_position,
    bump,
)
from conf.usr_conf import usr_conf_check
from diy.config import Config
from types import SimpleNamespace
import random
import time

@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]

    epoch_num = 100000
    episode_num_every_epoch = 100
    g_data_truncat = 256
    last_save_model_time = 0

    random.seed(time.time())

    roads=[(0, 6, 0), (0, 9, 0), (0, 10, 0), (2, 15, 0), (3, 6, 0), (3, 7, 0), (3, 11, 0), (4, 11, 0), (5, 12, 0), (6, 0, 0), (6, 3, 0), (6, 7, 0), (6, 9, 0), (6, 10, 0), (6, 11, 0), (7, 3, 0), (7, 6, 0), (7, 11, 0), (8, 9, 0), (8, 13, 0), (9, 0, 0), (9, 8, 0), (9, 10, 0), (9, 12, 0), (9, 14, 0), (10, 0, 0), (10, 6, 0), (10, 9, 0), (10, 14, 0), (10, 15, 0), (11, 3, 0), (11, 4, 0), (11, 6, 0), (11, 7, 0), (12, 5, 0), (12, 9, 0), (12, 14, 0), (13, 1, 0), (13, 8, 0), (13, 9, 0), (14, 9, 0), (14, 10, 0), (14, 12, 0), (14, 15, 0), (15, 2, 0), (15, 10, 0), (15, 14, 0), (2, 7, 1), (2, 15, 1), (4, 13, 1), (5, 8, 1)]

    index = random.randint(0, len(roads)-1)

    # start = roads[index][0]
    # end = roads[index][1]
    # can_flicker = roads[index][2]
    start = 5
    current_end = 8
    can_flicker = 1

    # User-defined game start configuration
    # 用户自定义的游戏启动配置
    usr_conf = {
        "diy": {
            "start": start,
            "end": 1,
            "current_end": current_end,
            "treasure_id": [current_end],
            "can_flicker": can_flicker,
            "treasure_random": 0,
            "talent_type": 1,
            "treasure_num": 1,
            "max_step": 200,
        }
    }
    logger.info(f"usr_conf start:{usr_conf['diy']['start']}, end:{usr_conf['diy']['end']}, treasure_num:{usr_conf['diy']['treasure_num']}")

    # logger.info(f"usr_conf  start:{usr_conf["diy"]["start"]}, end:{usr_conf["diy"]["end"]}, treasure_num:{usr_conf["diy"]["treasure_num"]}")
    # usr_conf_check is a tool to check whether the game configuration is correct
    # It is recommended to perform a check before calling reset.env
    # usr_conf_check会检查游戏配置是否正确，建议调用reset.env前先检查一下

    valid = usr_conf_check(usr_conf, logger)
    if not valid:
        logger.error(f"usr_conf_check return False, please check")
        return


    for epoch in range(epoch_num):
        epoch_total_rew = 0
        data_length = 0
        for g_data in run_episodes(episode_num_every_epoch, env, agent, g_data_truncat, usr_conf, logger, monitor):
            logger.info(f"len(g_data): {len(g_data)}")
            data_length += len(g_data)
            total_rew = sum([i.rew for i in g_data])
            epoch_total_rew += total_rew
            agent.learn(g_data)
            g_data.clear()

        avg_step_reward = 0
        if data_length:
            avg_step_reward = f"{(epoch_total_rew/data_length):.2f}"

        # save model file
        # 保存model文件
        now = time.time()
        if now - last_save_model_time >= 120:
            agent.save_model()
            last_save_model_time = now

        logger.info(f"Avg Step Reward: {avg_step_reward}, Epoch: {epoch}, Data Length: {data_length}")


    # At the start of each game, support loading the latest model file
    # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
    # agent.load_model(id="latest")
    
    # model saving
    # 保存模型
    # agent.save_model()

    return

def get_current_end_xz(current_end):
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
    norm_x = data_dict[current_end]["norm_x"]
    norm_z = data_dict[current_end]["norm_z"]
    grid_x = data_dict[current_end]["grid_x"]
    grid_z = data_dict[current_end]["grid_z"]
    return norm_x,norm_z,grid_x,grid_z

def modify_obs(obs, start, current_end, need_to_collect_buff, can_use_talent):
    norm_x,norm_z,grid_x,grid_z = get_current_end_xz(current_end)

    grid_pos = SimpleNamespace(**{'x':grid_x,'z':grid_z})
    one_hot_pos = one_hot_encoding(grid_pos)
    obs.feature[260] = norm_x
    obs.feature[261] = norm_z
    obs.feature[262] = grid_x / 128
    obs.feature[263] = grid_z / 128
    for i in range(128*2):
        obs.feature[i+264] = one_hot_pos[i]
    obs.feature[520] = need_to_collect_buff
    obs.feature[521] = can_use_talent #用来训练闪现的 这里先设为0，先不训
    if can_use_talent:
        obs.legal_act = [1,1]
    else:
        obs.legal_act = [1,0]

    return obs


def check_obstacle(obs_data, act):
    # Extract the last 51x51 elements as the obstacle map
    obstacle_list = obs_data.feature[-51*51:]
    
    # Convert the 51x51 list into a 2D numpy array
    obstacle_map = np.array(obstacle_list).reshape(51, 51)
    
    directions = [
        (1, 0),   # Right
        (1, -1),  # Top-right
        (0, -1),  # Up
        (-1, -1), # Top-left
        (-1, 0),  # Left
        (-1, 1),  # Bottom-left
        (0, 1),   # Down
        (1, 1)    # Bottom-right
    ]
    
    size = obstacle_map.shape[0]  # Assuming square map (51x51)
    center_pos = (size // 2, size // 2)
    
    # Perform modulo 8 on the input act to determine the direction
    direction_index = act % 8
    dx, dy = directions[direction_index]
    
    x_start, y_start = center_pos
    
    # Check the 16-cell range in the determined direction
    for i in range(1, 17):
        if dx != 0 and dy != 0:  # For diagonal directions
            x = x_start + round(i * dx / np.sqrt(2))
            y = y_start + round(i * dy / np.sqrt(2))
        else:
            x = x_start + i * dx
            y = y_start + i * dy
        
        # Ensure the new position is within bounds
        if 0 <= x < size and 0 <= y < size:
            if obstacle_map[y, x] == 1:  # Assuming obstacle cells are marked with 1
                return True
        else:
            break  # Stop if out of bounds
            
    return False

def run_episodes(n_episode, env, agent, g_data_truncat, usr_conf, logger,monitor):
    for episode in range(n_episode):
        collector = list()

        # Reset the game and get the initial state
        # 重置游戏, 并获取初始状态
        obs = env.reset(usr_conf=usr_conf)
        start = usr_conf['diy']['start']
        current_end = usr_conf['diy']['current_end']
        env_info = None

        # Disaster recovery
        # 容灾
        if obs is None:
            continue

        # At the start of each game, support loading the latest model file
        # The call will load the latest model from a remote training node
        # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
        
        # debug
        if Config.DEBUG != True:
            agent.load_model(id="latest")

        # Feature processing
        # 特征处理
        ####################
        need_to_collect_buff = False
        if (start,current_end) in [(6,9),(9,6),(6,10),(10,6),(9,10),(10,9)]:
            need_to_collect_buff = True
        collect_buff_cnt = 0

        can_use_talent = False
        if (start,current_end) in [(4,13),(5,8)]:
            can_use_talent = True
        use_talent_cnt = 0
        reach_current_end = False
        ####################


        obs_data = observation_process(obs)
        modify_obs(obs_data, usr_conf['diy']['start'], current_end, need_to_collect_buff,can_use_talent)

        done = False
        step = 0
        bump_cnt = 0

        while not done:
            # Agent performs inference, gets the predicted action for the next frame
            # Agent 进行推理, 获取下一帧的预测动作
            # 增加了一个prob_data返回值，它的格式与act_data保持一致
            act_data = agent.predict(list_obs_data=[obs_data])

            act_data = act_data[0]
            act = action_process(act_data)

            prob = act_data.prob

            # Interact with the environment, execute actions, get the next state
            # 与环境交互, 执行动作, 获取下一步的状态
            frame_no, _obs, score, terminated, truncated, _env_info = env.step(act)
            if _obs is None:
                break
            

            step += 1

            # Feature processing
            # 特征处理
            _obs_data = observation_process(_obs)
            modify_obs(_obs_data, usr_conf['diy']['start'], current_end, need_to_collect_buff, can_use_talent)

            current_end_norm_x = obs_data.feature[260]
            current_end_norm_z = obs_data.feature[261]
            current_end_grid_x = obs_data.feature[262]
            current_end_grid_z = obs_data.feature[263]

            pre_norm_x = obs_data.feature[0]
            pre_norm_z = obs_data.feature[1]
            pre_grid_x = obs_data.feature[2]
            pre_grid_z = obs_data.feature[3]

            now_norm_x = _obs_data.feature[0]
            now_norm_z = _obs_data.feature[1]
            now_grid_x = _obs_data.feature[2]
            now_grid_z = _obs_data.feature[3]

            end_dist = ((current_end_grid_x - now_grid_x)**2 + (current_end_grid_z - now_grid_z)**2) ** 1/2.0
            move_dist = ((pre_norm_x - now_norm_x)**2 + (pre_norm_z - now_norm_z)**2) ** 1/2.0
            use_talent = False
            if end_dist<=1/128: #约等于500码
                reach_current_end = True
            if act>=8:
                use_talent = True
                can_use_talent = False
            super_flicker_type = -1 #未使用超级闪现
            if use_talent:
                if check_obstacle(obs_data,act) and move_dist>0.125: #如果是超级闪现
                    super_flicker_type = 1
                else:
                    super_flicker_type = 0 #若使用了并且失败了
            
            buff_dist = ((current_end_grid_x - 59)**2 + (current_end_grid_z - 53)**2) ** 1/2.0
            if buff_dist<=1/128: #约等于500码
                collect_buff_cnt += 1

            # Disaster recovery
            # 容灾
            if truncated and frame_no is None:
                break

            treasures_num = 0
            # Calculate reward
            # 计算 reward
            if env_info is None:
                reward = 0
            else:
                reward, is_bump = reward_shaping(
                    obs_data,
                    act,
                    _obs_data,
                    logger,
                    need_to_collect_buff,
                    collect_buff_cnt,
                    super_flicker_type,
                    move_dist,
                    reach_current_end
                )

                treasure_dists = [pos.grid_distance for pos in _obs.feature.treasure_pos]
                treasures_num = treasure_dists.count(1.0)

                # Wall bump behavior statistics
                # 撞墙行为统计
                bump_cnt += is_bump

            # Determine game over, and update the number of victories
            # 判断游戏结束, 并更新胜利次数
            if truncated:
                logger.info(
                    f"truncated is True, so this episode {episode} timeout, \
                        collected treasures: {treasures_num  - 7}"
                )
            elif terminated:
                logger.info(
                    f"terminated is True, so this episode {episode} reach the end, \
                        collected treasures: {treasures_num  - 7}"
                )

            done = terminated or truncated or reach_current_end or (super_flicker_type == -1) #若闪现失败了也退出

            # Construct game frames to prepare for sample construction
            # 构造游戏帧，为构造样本做准备

            modify_obs(_obs_data, usr_conf['diy']['start'], current_end, need_to_collect_buff, can_use_talent) 
            #这里之所以要modify两次，是因为要把can use talent变量重置进去，一开始为了方便没有弄进去。
            # 增加了一个prob，为PPO服务

            frame = Frame(
                obs=obs_data.feature,
                _obs=_obs_data.feature,
                obs_legal=obs_data.legal_act,
                _obs_legal=_obs_data.legal_act,
                act=act,
                prob=prob,
                rew=reward,
                done=done,
                ret=reward
            )
            
            collector.append(frame)

            # If the game is over, the sample is processed and sent to training
            # 如果游戏结束，则进行样本处理，将样本送去训练

            logger.info(f"len(collector): {len(collector)} (x,z):({now_grid_x * 128:.0f},{now_grid_z * 128:.0f}), act: {act}, dist:{move_dist*64000:.0f} norm_dist:{move_dist:.6f}, prob: {prob:.2f}, reward: {reward}")
            # logger.info(f"")
            if done:
                if len(collector) > 0:
                    return_to_go = 0
                    for frame in reversed(collector):
                        return_to_go = frame.rew + return_to_go * Config.GAMMA
                        frame.ret = return_to_go
                    if not(return_to_go > -5 and return_to_go < 20):
                        logger.info(f"return_to_go: {return_to_go}")
                    collector = sample_process(collector)
                    monitor_data = {
                        "diy_5": return_to_go,
                    }
                    if monitor:
                        monitor.put_data({os.getpid(): monitor_data})

                    yield collector
                break

            # Status update
            # 状态更新
            obs_data = _obs_data
            obs = _obs
            env_info = _env_info
