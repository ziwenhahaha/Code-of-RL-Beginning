import numpy as np
import random
from GridWorld_v1 import Animator, GridWorld_v1

# 跟v1版本的区别主要是两点，v1是针对deteministic的策略的，v2是针对stochastic的策略的，
# 具体来说的话就是，v2版本支持在同一个state概率选择若干个动作
# 它的策略矩阵，现在是 shape==(25,5)的第一维表示state，第二维表示action，返回一个概率
# 在打印策略的时候，将把每个state最大概率的动作打印出来
#
# 第二点区别是，在v2版本里面，引入了trajectory的概念
# 通过getTrajectoryScore方法可以直接按照提供的policy，进行采样若干步


class GridWorld_v2(GridWorld_v1):
    # n行，m列，随机若干个forbiddenArea，随机若干个target
    # A1: move upwards
    # A2: move rightwards;
    # A3: move downwards;
    # A4: move leftwards;
    # A5: stay unchanged;

    stateMap = None  # 大小为rows*columns的list，每个位置存的是state的编号
    scoreMap = None  # 大小为rows*columns的list，每个位置存的是奖励值 0 1 -1
    score = 0  # targetArea的得分
    forbiddenAreaScore = 0  # forbiddenArea的得分

    def get_trajectory_score(
        self,
        now_state: tuple[int, int],
        action: int,
        policy: np.ndarray,
        steps: int,
        stop_when_reach_target: bool = False,
    ) -> list[tuple[tuple[int, int], int, int, int, int]]:
        # policy是一个 (rows,columns, actions)的三维列表，其中每一行的总和为1，代表每个state选择五个action的概率总和为1
        # Attention: 返回值是一个大小为steps+1的列表，因为第一步也计算在里面了
        # 其中的元素是(nowState, nowAction, score, nextState, nextAction)元组

        res = []
        now_action = action
        if stop_when_reach_target is True:
            steps = 20000
        for i in range(steps + 1):

            score, next_state = self.get_score(now_state, now_action)
            next_action = np.random.choice(
                range(5), size=1, replace=False, p=policy[next_state]
            )[0]

            res.append((now_state, now_action, score, next_state, next_action))
            now_state = next_state
            now_action = next_action

            if stop_when_reach_target:
                # print(nextState)
                # print(self.scoreMap)
                nowx = now_state // self.columns
                nowy = now_state % self.columns
                if self.scoreMap[nowx][nowy] == self.score:
                    return res
        return res
