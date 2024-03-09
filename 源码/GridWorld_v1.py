import numpy as np
import random
class GridWorld_v1(object): 
    # 初版gridworld，没有写trajectory逻辑以及，policy维度仅为1*25，
    # 目的是用来计算非stochastic情况下policy iteration和value iteration 的贝尔曼方程解

    # n行，m列，随机若干个forbiddenArea，随机若干个target
    # A1: move upwards
    # A2: move rightwards;
    # A3: move downwards;
    # A4: move leftwards;
    # A5: stay unchanged;

    stateMap = None  #大小为rows*columns的list，每个位置存的是state的编号   
    scoreMap = None  #大小为rows*columns的list，每个位置存的是奖励值 0 1 -10
    score = 0             #targetArea的得分
    forbiddenAreaScore=0  #forbiddenArea的得分

    
    def __init__(self,rows=4, columns=5, forbiddenAreaNums=3, targetNums=1, seed = -1, score = 1, forbiddenAreaScore = -1, desc=None):
        #1、构造函数（构造一个自定义or随机的网格世界）
        self.score = score
        self.forbiddenAreaScore = forbiddenAreaScore
        if(desc != None):
            #if the gridWorld is fixed
            self.rows = len(desc)
            self.columns = len(desc[0])
            l = []
            for i in range(self.rows):
                tmp = []
                for j in range(self.columns):
                    tmp.append(forbiddenAreaScore if desc[i][j]=='#' else score if desc[i][j]=='T' else 0)
                l.append(tmp)
            self.scoreMap = np.array(l)
            self.stateMap = [[i*self.columns+j for j in range(self.columns)] for i in range(self.rows)]
            return
            
        #if the gridWorld is random
        self.rows = rows
        self.columns = columns
        self.forbiddenAreaNums = forbiddenAreaNums
        self.targetNums = targetNums
        self.seed = seed

        random.seed(self.seed)
        l = [i for i in range(self.rows * self.columns)]
        random.shuffle(l)  #用shuffle来重排列
        self.g = [0 for i in range(self.rows * self.columns)]
        for i in range(forbiddenAreaNums):
            self.g[l[i]] = forbiddenAreaScore;        # 设置禁止进入的区域，惩罚为1
        for i in range(targetNums):
            self.g[l[forbiddenAreaNums+i]] = score # 奖励值为1的targetArea
            
        self.scoreMap = np.array(self.g).reshape(rows,columns)
        self.stateMap = [[i*self.columns+j for j in range(self.columns)] for i in range(self.rows)]

    def show(self):
        #2、把网格世界展示出来（show函数）
        for i in range(self.rows):
            s = ""
            for j in range(self.columns):
                tmp = {0:"⬜️",self.forbiddenAreaScore:"🚫",self.score:"✅"}
                s = s + tmp[self.scoreMap[i][j]]
            print(s)

    #5*5
    def getScore(self, nowState, action):
        #3、在当前状态[0,24]，执行动作[0,4]的得分及下一个状态
        nowx = nowState // self.columns
        nowy = nowState % self.columns
        
        if(nowx<0 or nowy<0 or nowx>=self.rows or nowy>=self.columns):
            print(f"coordinate error: ({nowx},{nowy})")
        if(action<0 or action>=5 ):
            print(f"action error: ({action})")
            
        # 上右下左 不动
        actionList = [(-1,0),(0,1),(1,0),(0,-1),(0,0)]
        tmpx = nowx + actionList[action][0]
        tmpy = nowy + actionList[action][1]
        # print(tmpx,tmpy)
        if(tmpx<0 or tmpy<0 or tmpx>=self.rows or tmpy>=self.columns):
            return -1,nowState
        return self.scoreMap[tmpx][tmpy],self.stateMap[tmpx][tmpy]

    def showPolicy(self, policy):
        #4、把传递进来的policy参数，进行可视化展示
        #用emoji表情，可视化策略，在平常的可通过区域就用普通箭头⬆️➡️⬇️⬅️
        #但若是forbiddenArea，那就十万火急急急,于是变成了双箭头⏫︎⏩️⏬⏪
        rows = self.rows
        columns = self.columns
        s = ""
        for i in range(self.rows * self.columns):
            nowx = i // columns
            nowy = i % columns
            if(self.scoreMap[nowx][nowy]==self.score):
                s = s + "✅"
            if(self.scoreMap[nowx][nowy]==0):
                tmp = {0:"⬆️",1:"➡️",2:"⬇️",3:"⬅️",4:"🔄"}
                s = s + tmp[policy[i]]
            if(self.scoreMap[nowx][nowy]==self.forbiddenAreaScore):
                tmp = {0:"⏫️",1:"⏩️",2:"⏬",3:"⏪",4:"🔄"}
                s = s + tmp[policy[i]]
            if(nowy == columns-1):
                print(s)
                s = ""