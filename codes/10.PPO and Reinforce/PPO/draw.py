import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import numpy as np
def draw(state_value, policy):
    x = np.linspace(0,5,5)
    y = np.linspace(0,5,5)
    
    X, Y = np.meshgrid(x,y)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection = '3d')
    ax1.plot_surface(Y,X,state_value, cmap='summer')
    ax1.set_title('3D Plot')

    ax2 = fig.add_subplot(122)
    im = ax2.imshow(state_value, cmap='summer',origin='upper')
    ax2.set_title('2D Heatmap')

    for i in range(len(X)):
        for j in range(len(Y)):
            text = ax2.text(j,i, round(state_value[i][j], 2),ha='center',va='center',color='black',alpha=1,fontsize=8)

    logos = ['↑','→','↓','←','○']

    for i in range(len(X)):
        for j in range(len(Y)):
            text = ax2.text(j,i, logos[policy[i*5+j]],ha='center',va='center',color='pink',alpha=0.6,fontsize=30)
    plt.colorbar(im)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# 示例概率数据，假设4x4的Gridworld
# now_frame_probabilities = np.random.rand(4, 4, 4)  # 随机生成每个方向的概率


def plot_policy(pre_frame_probabilities,
                now_frame_probabilities, 
                trajectory_state_action_score = np.zeros((25,4)), 
                transition_list = [], # SARS`
                mpdesc = [".....",".##..","..#..",".#T#.",".#..."], 
                
                img_path = None):

    mp = mpdesc
    
    rows = len(mp)
    columns = len(mp[0])
    
    # states_trajectory.append([3,2]) # 把最后目的地加入进去，方面画图
    
    """
        下面这段是测试代码
        now_frame_probabilities = np.random.rand(25, 4)  # 随机生成每个方向的概率
        pre_frame_probabilities = np.random.rand(25, 4)  # 随机生成每个方向的概率
        trajectory_state_action_score = np.zeros((25,4))  # 初始化每个方向的得分
        
        mpdesc = [".....",".##..","..#..",".#T#.",".#..."]
        states = [[1,2],[1,3]]
        trajectory_state_action_score[7][1] = -9.7
        trajectory_state_action_score[8][2] = 30
        
        trajectory_state_action_score[9][2] = -9.9
        
        # img_path 
        plot_policy(pre_frame_probabilities, now_frame_probabilities,trajectory_state_action_score, states, mpdesc, img_path=None)
    """

    
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(11, 4))

    ax_pre_frame_probabilities = axs[0]
    ax_trajectory = axs[1]
    ax_now_frame_probabilities = axs[2]

    ax_pre_frame_probabilities.set_title('pre frame', fontproperties='DejaVu Sans', fontsize=10)
    ax_trajectory.set_title('trajectory', fontproperties='DejaVu Sans', fontsize=10)
    ax_now_frame_probabilities.set_title('now frame', fontproperties='DejaVu Sans', fontsize=10)

    #########################################################################################################################
    ##############################################      上一帧的概率可视化      ###############################################
    #########################################################################################################################

    arrow_width = 1
    actions = None
    directions = None
    offset = None
    offset_arraw = None  #箭头的偏移
    offset_text = None  
    
    if len(pre_frame_probabilities[0])==5:
        actions = ['up', 'right', 'down', 'left', 'stay']
        offset_arraw = 0.1
        offset_text = 0.1
        directions = {'up': (0, 0.7), 'right': (0.7, 0), 'down': (0, -0.7), 'left': (-0.7, 0), 'stay': (0, 0)}
    else:
        actions = ['up', 'right', 'down', 'left']
        offset_arraw = 0.04
        offset_text = 0.06
        offset = 0.0
        
        directions = {'up': (0, 0.8), 'right': (0.8, 0), 'down': (0, -0.8), 'left': (-0.8, 0), 'stay': (0, 0)}
    # 绘制网格

    mp = mpdesc
    for i in range(rows):
        for j in range(columns):
            if mp[i][j] == '.':
                color = 'white'
            if mp[i][j] == '#':
                color = '#F5C142'
            if mp[i][j] == 'T':
                color = '#65FFFF'  
            ax_pre_frame_probabilities.add_patch(plt.Rectangle((j, rows -1 - i), 1, 1, facecolor=color, edgecolor='black', linewidth=0.2))
            
    for x in range(6):
        ax_pre_frame_probabilities.axhline(x, lw=0.2, color='black', zorder=0)
        ax_pre_frame_probabilities.axvline(x, lw=0.2, color='black', zorder=0)
        
    
    offsets_arraw = {
    'up': (0, offset_arraw),
    'right': (offset_arraw, 0),
    'down': (0, -offset_arraw),
    'left': (-offset_arraw, 0),
    'stay': (0, 0)
    }
    offsets_text = {
    'up': (0, offset_text),
    'right': (offset_text, 0),
    'down': (0, -offset_text),
    'left': (-offset_text, 0),
    'stay': (0, 0)
    }
    # 绘制箭头
    for i in range(rows):
        for j in range(columns):
            for k, action in enumerate(actions):
                dx, dy = directions[action]
                
                prob = pre_frame_probabilities[i*columns+j][k]
                
                ox_a, oy_a = offsets_arraw[action]
                ox_t, oy_t = offsets_text[action]

                color = "gray"
                arrowstyle = "-"
                len_scale = 0.3
                if k == np.array(pre_frame_probabilities[i*columns+j]).argmax():
                    color = "red"
                    arrowstyle = '->'
                    len_scale = 0.37
                
                if action == 'stay':
                    # 用圆圈表示不动的动作
                    circle = plt.Circle((j + 0.5 , rows-0.5  - i ), 0.1, color=color, fill=False, linewidth=1)
                    ax_pre_frame_probabilities.add_patch(circle)
                    ax_pre_frame_probabilities.text(j + 0.5 + ox, rows-0.5  - i + oy, f'{prob*100:.0f}', color=color, ha='center', va='center', fontsize=7)
                
                arrow = FancyArrowPatch((j + 0.5+ ox_a, rows-1 - i + 0.5+ oy_a), (j + 0.5 + dx * len_scale, rows-1 - i + 0.5 + dy * len_scale),
                                        arrowstyle=arrowstyle, mutation_scale=5, lw=arrow_width, color=color)
                ax_pre_frame_probabilities.add_patch(arrow)
                ax_pre_frame_probabilities.text(j + 0.5 + ox_a + dx * 0.3+ox_t, rows-0.5  - i + oy_a + dy * 0.3+oy_t, f'{prob*100:.0f}', color=color, ha='center', va='center', fontsize=7)
                
    ax_pre_frame_probabilities.set_xlim(-0.05, columns + 0.05)
    ax_pre_frame_probabilities.set_ylim(-0.05, rows + 0.05)
    ax_pre_frame_probabilities.set_aspect('equal')
    ax_pre_frame_probabilities.axis('off')



    #########################################################################################################################
    ##########################################           轨迹绘制             ################################################
    #########################################################################################################################
    
    arrow_width = 1
    actions = ['up', 'right', 'down', 'left', 'stay']
    directions = {'up': (0, 0.7), 'right': (0.7, 0), 'down': (0, -0.7), 'left': (-0.7, 0), 'stay': (0, 0)}
    # 绘制网格

    for i in range(rows):
        for j in range(columns):
            if mp[i][j] == '.':
                color = 'white'
            if mp[i][j] == '#':
                color = '#F5C142'
            if mp[i][j] == 'T':
                color = '#65FFFF'  
            ax_trajectory.add_patch(plt.Rectangle((j, rows-1 - i), 1, 1, facecolor=color, edgecolor='black', linewidth=0.2))
            
    for x in range(6):
        ax_trajectory.axhline(x, lw=0.2, color='black', zorder=0)
        ax_trajectory.axvline(x, lw=0.2, color='black', zorder=0)

    left = min(len(transition_list) ,250)
    for i in range(left):
        start = transition_list[i][0]
        start = (start[1],rows-1-start[0])
        end = transition_list[i][3]
        end = (end[1],rows-1-end[0])
        dx = end[0] - start[0] + np.random.uniform(-0.15, 0.15)  # 添加随机偏移
        dy = end[1] - start[1] + np.random.uniform(-0.15, 0.15)  # 添加随机偏移
        alpha = 0.2
        color_intensity = np.clip(alpha, 0, 1)
        # color = (color_intensity, 0, 1 - color_intensity)  # 颜色从浅蓝变为深蓝
        color = "black"
        ax_trajectory.arrow(start[0]+0.5, start[1]+0.5, dx, dy, head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.2)
    
    for i in range(max(len(transition_list)-250, left), len(transition_list)):
        start = transition_list[i][0]
        start = (start[1],rows-1-start[0])
        end = transition_list[i][3]
        end = (end[1],rows-1-end[0])
        dx = end[0] - start[0] + np.random.uniform(-0.15, 0.15)  # 添加随机偏移
        dy = end[1] - start[1] + np.random.uniform(-0.15, 0.15)  # 添加随机偏移
        alpha = 0.2
        color_intensity = np.clip(alpha, 0, 1)
        # color = (color_intensity, 0, 1 - color_intensity)  # 颜色从浅蓝变为深蓝
        color = "black"
        ax_trajectory.arrow(start[0]+0.5, start[1]+0.5, dx, dy, head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.2)




    arrow_width = 1
    actions = None
    directions = None
    offset = None
    offset_arraw = None  #箭头的偏移
    offset_text = None  
    
    if len(now_frame_probabilities[0])==5:
        actions = ['up', 'right', 'down', 'left', 'stay']
        offset_arraw = 0.1
        offset_text = 0.1
        directions = {'up': (0, 0.7), 'right': (0.7, 0), 'down': (0, -0.7), 'left': (-0.7, 0), 'stay': (0, 0)}
    else:
        actions = ['up', 'right', 'down', 'left']
        offset_arraw = 0.04
        offset_text = 0.06
        offset = 0.0
        
        directions = {'up': (0, 0.8), 'right': (0.8, 0), 'down': (0, -0.8), 'left': (-0.8, 0), 'stay': (0, 0)}

    
    offsets_arraw = {
    'up': (0, offset_arraw),
    'right': (offset_arraw, 0),
    'down': (0, -offset_arraw),
    'left': (-offset_arraw, 0),
    'stay': (0, 0)
    }
    offsets_text = {
    'up': (0, offset_text),
    'right': (offset_text, 0),
    'down': (0, -offset_text),
    'left': (-offset_text, 0),
    'stay': (0, 0)
    }
    # 绘制箭头
    for i in range(rows):
        for j in range(columns):
            for k, action in enumerate(actions):
                dx, dy = directions[action]
                
                score = trajectory_state_action_score[i*columns+j][k]
                
                ox_a, oy_a = offsets_arraw[action]
                ox_t, oy_t = offsets_text[action]

                color = "gray"
                arrowstyle = "-"
                len_scale = 0.3
                
                if score == 0:
                    continue

                text_str = ""
                score = max(-99,score)
                score = min( 99,score)
                
                
                if score > 0:
                    color = "red"
                    arrowstyle = '->'
                    len_scale = 0.37
                    if score>10:
                        text_str = f'{score:.0f}'
                    else:
                        text_str = f'{score:.1f}'
                    
                if score < 0:
                    score = -score
                    color = "blue"
                    arrowstyle = '->'
                    len_scale = 0.37
                    if score>10:
                        text_str = f'{score:.0f}'
                    else:
                        text_str = f'{score:.1f}'
                
                if action == 'stay':
                    # 用圆圈表示不动的动作
                    circle = plt.Circle((j + 0.5 , rows-0.5  - i ), 0.1, color=color, fill=False, linewidth=1)
                    # ax_trajectory.add_patch(circle)
                    ax_trajectory.text(j + 0.5 + ox, rows-0.5  - i + oy, text_str, color=color, ha='center', va='center', fontsize=6, alpha=1)
                
                arrow = FancyArrowPatch((j + 0.5+ ox_a, rows-1 - i + 0.5+ oy_a), (j + 0.5 + dx * len_scale, rows-1 - i + 0.5 + dy * len_scale),
                                        arrowstyle=arrowstyle, mutation_scale=5, lw=arrow_width, color=color, alpha=0.7)
                ax_trajectory.add_patch(arrow)
                ax_trajectory.text(j + 0.5 + ox_a + dx * 0.3+ox_t, rows-0.5  - i + oy_a + dy * 0.3+oy_t, text_str, color=color, ha='center', va='center', fontsize=6, alpha=1)
                
    ax_trajectory.set_xlim(-0.05, columns + 0.05)
    ax_trajectory.set_ylim(-0.05, rows + 0.05)
    ax_trajectory.set_aspect('equal')
    ax_trajectory.axis('off')

    



    #########################################################################################################################
    ##############################################      当前帧的概率可视化      ###############################################
    #########################################################################################################################

    arrow_width = 1
    actions = None
    directions = None
    offset = None
    offset_arraw = None  #箭头的偏移
    offset_text = None  
    
    if len(now_frame_probabilities[0])==5:
        actions = ['up', 'right', 'down', 'left', 'stay']
        offset_arraw = 0.1
        offset_text = 0.1
        directions = {'up': (0, 0.7), 'right': (0.7, 0), 'down': (0, -0.7), 'left': (-0.7, 0), 'stay': (0, 0)}
    else:
        actions = ['up', 'right', 'down', 'left']
        offset_arraw = 0.04
        offset_text = 0.06
        offset = 0.0
        
        directions = {'up': (0, 0.8), 'right': (0.8, 0), 'down': (0, -0.8), 'left': (-0.8, 0), 'stay': (0, 0)}
    # 绘制网格

    mp = mpdesc
    for i in range(rows):
        for j in range(columns):
            if mp[i][j] == '.':
                color = 'white'
            if mp[i][j] == '#':
                color = '#F5C142'
            if mp[i][j] == 'T':
                color = '#65FFFF'  
            ax_now_frame_probabilities.add_patch(plt.Rectangle((j, rows -1 - i), 1, 1, facecolor=color, edgecolor='black', linewidth=0.2))
            
    for x in range(6):
        ax_now_frame_probabilities.axhline(x, lw=0.2, color='black', zorder=0)
        ax_now_frame_probabilities.axvline(x, lw=0.2, color='black', zorder=0)
        
    
    offsets_arraw = {
    'up': (0, offset_arraw),
    'right': (offset_arraw, 0),
    'down': (0, -offset_arraw),
    'left': (-offset_arraw, 0),
    'stay': (0, 0)
    }
    offsets_text = {
    'up': (0, offset_text),
    'right': (offset_text, 0),
    'down': (0, -offset_text),
    'left': (-offset_text, 0),
    'stay': (0, 0)
    }
    # 绘制箭头
    for i in range(rows):
        for j in range(columns):
            for k, action in enumerate(actions):
                dx, dy = directions[action]
                
                prob = now_frame_probabilities[i*columns+j][k]
                
                ox_a, oy_a = offsets_arraw[action]
                ox_t, oy_t = offsets_text[action]

                color = "gray"
                arrowstyle = "-"
                len_scale = 0.3
                if k == np.array(now_frame_probabilities[i*columns+j]).argmax():
                    color = "red"
                    arrowstyle = '->'
                    len_scale = 0.37
                
                if action == 'stay':
                    # 用圆圈表示不动的动作
                    circle = plt.Circle((j + 0.5 , rows-0.5  - i ), 0.1, color=color, fill=False, linewidth=1)
                    ax_now_frame_probabilities.add_patch(circle)
                    ax_now_frame_probabilities.text(j + 0.5 + ox, rows-0.5  - i + oy, f'{prob*100:.0f}', color=color, ha='center', va='center', fontsize=7)
                
                arrow = FancyArrowPatch((j + 0.5+ ox_a, rows-0.5 - i + oy_a), (j + 0.5 + dx * len_scale, rows-0.5  - i  + dy * len_scale),
                                        arrowstyle=arrowstyle, mutation_scale=5, lw=arrow_width, color=color)
                ax_now_frame_probabilities.add_patch(arrow)
                ax_now_frame_probabilities.text(j + 0.5 + ox_a + dx * 0.3+ox_t, rows-0.5  - i + oy_a + dy * 0.3+oy_t, f'{prob*100:.0f}', color=color, ha='center', va='center', fontsize=7)
                
    ax_now_frame_probabilities.set_xlim(-0.05, columns + 0.05)
    ax_now_frame_probabilities.set_ylim(-0.05, rows + 0.05)
    ax_now_frame_probabilities.set_aspect('equal')
    ax_now_frame_probabilities.axis('off')

    

    
    if img_path!=None:
        plt.savefig(img_path)
    plt.show()
    


