import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, clear_output


class Animator:
    def __init__(self, row: int, columes: int):
        self.rows = row
        self.columes = columes
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(3.5, 3.5)
        for row in range(row + 1):
            self.ax.plot([0, columes], [row, row], color="black")
        for col in range(columes + 1):
            self.ax.plot([col, col], [0, row], color="black")

        # 设置轴的范围
        self.ax.set_xlim(0, columes)
        self.ax.set_ylim(row, 0)

        # 禁用轴刻度
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.elements = []

    def add_rect(self, positions: list[tuple[int, int]], color: str):
        for x, y in positions:
            rect = patches.Rectangle(
                (y, x),
                1,
                1,
                linewidth=1,
                edgecolor="black",
                facecolor=color,
            )
            self.ax.add_patch(rect)

    def add_arrow(
        self,
        positions: list[tuple[int, int]],
        direction: tuple[int, int],
        color: str,
    ):
        for x, y in positions:
            dy, dx = direction
            arrow = patches.FancyArrowPatch(
                (y + 0.5, x + 0.5),
                (y + 0.5 + dy, x + 0.5 + dx),
                mutation_scale=10,
                arrowstyle="-|>",
                shrinkA=0,
                shrinkB=0,
                color=color,
            )
            self.ax.add_patch(arrow)
            self.elements.append(arrow)

    def add_circle(self, positions: list[tuple[int, int]], color: str = "#04B153"):
        for x, y in positions:
            circle = patches.Circle(
                (y + 0.5, x + 0.5),
                0.1,
                linewidth=1,
                edgecolor=color,
                facecolor="none",
            )
            self.ax.add_patch(circle)
            self.elements.append(circle)

    def add_value(self, value: np.ndarray, color: str = "black"):
        # for value, positions in zip(value, np.ndindex(value.shape)):
        for position in np.ndindex(value.shape):
            value_elements = self.ax.text(
                position[1] + 0.5,
                position[0] + 0.5,
                f"{value[position]:.1f}",
                color=color,
                fontsize=10,
                ha="center",
                va="center",
            )
            self.elements.append(value_elements)

    def clear_elements(self):
        for element in self.elements:
            element.remove()
        self.elements.clear()


class GridWorld_v1(object):
    # 初版gridworld，没有写trajectory逻辑以及，policy维度仅为1*25，
    # 目的是用来计算非stochastic情况下policy iteration和value iteration 的贝尔曼方程解

    # n行，m列，随机若干个forbiddenArea，随机若干个target
    # A1: move upwards
    # A2: move rightwards;
    # A3: move downwards;
    # A4: move leftwards;
    # A5: stay unchanged;

    state_map = None  # 大小为rows*columns的list，每个位置存的是state的编号
    score_map = None  # 大小为rows*columns的list，每个位置存的是奖励值 0 1 -10
    score = 0  # targetArea的得分
    forbidden_area_score = 0  # forbiddenArea的得分

    def __init__(
        self,
        rows: int = 4,
        columns: int = 5,
        forbidden_area_nums: int = 3,
        target_nums: int = 1,
        seed: int = -1,
        score: int = 1,
        forbidden_area_score: int = -1,
        desc: list = None,
    ):
        # 1、构造函数（构造一个自定义or随机的网格世界）
        self.score = score
        self.forbidden_area_score = forbidden_area_score
        self.forbidden_area_nums = forbidden_area_nums
        self.target_nums = target_nums
        self.seed = seed
        if desc is not None:
            # if the gridWorld is fixed
            self.rows = len(desc)
            self.columns = len(desc[0])
            self.score_map = np.zeros((self.rows, self.columns))
            desc_array = np.array(
                [list(row) for row in desc]
            )  # change the desc to a 2D numpy array
            self.score_map[desc_array == "#"] = forbidden_area_score
            self.score_map[desc_array == "T"] = score
        else:
            # generate a random grid_world
            self.rows = rows
            self.columns = columns
            random.seed(self.seed)
            self.score_map = np.zeros((self.rows, self.columns))
            forbidden_target_idx = random.sample(
                range(self.rows * self.columns), forbidden_area_nums + target_nums
            )
            forbidden_idx, target_idx = (
                forbidden_target_idx[:forbidden_area_nums],
                forbidden_target_idx[forbidden_area_nums:],
            )
            forbidden_idx = random.sample(
                range(self.rows * self.columns), forbidden_area_nums
            )
            self.score_map.flat[forbidden_idx] = forbidden_area_score
            self.score_map.flat[target_idx] = score

        self.state_map = np.arange(rows * columns).reshape(rows, columns)
        self.animator = Animator(self.rows, self.columns)

    def show(self):
        # 2、把网格世界展示出来（show函数）

        target_positions = list(zip(*np.where(self.score_map == self.score)))
        forbiden_positions = list(
            zip(*np.where(self.score_map == self.forbidden_area_score))
        )
        # used to visualizes the grid world by showing the target positions and forbidden positions on the grid
        self.animator.add_rect(target_positions, "#65FFFF")
        self.animator.add_rect(forbiden_positions, "#FFBF01")
        self.animator.clear_elements()
        display(self.animator.fig)
        clear_output(wait=True)
        # plt.show()

    # 5*5
    def get_score(
        self, nowState: tuple[int, int], action: int
    ) -> tuple[int, tuple[int, int]]:
        # 3、在当前状态[0,24]，执行动作[0,4]的得分及下一个状态
        now_x, now_y = nowState
        if now_x < 0 or now_y < 0 or now_x >= self.rows or now_y >= self.columns:
            print(f"coordinate error: ({now_x},{now_y})")
        if action < 0 or action >= 5:
            print(f"action error: ({action})")

        # 上右下左 不动
        actionList = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
        next_x = now_x + actionList[action][0]
        next_y = now_y + actionList[action][1]
        if next_x < 0 or next_y < 0 or next_x >= self.rows or next_y >= self.columns:
            return -1, nowState
        return self.score_map[next_x][next_y], (next_x, next_y)

    def show_value(self, value: np.ndarray, step: bool = True):
        # to show q value or state value
        self.animator.clear_elements()
        self.animator.add_value(value)
        if not step:
            clear_output(wait=True)
        display(self.animator.fig)

    def show_policy(self, policy: np.ndarray, step: bool = True):
        arrow = {
            "up": (0, -0.5),
            "right": (0.5, 0),
            "down": (0, 0.5),
            "left": (-0.5, 0),
        }
        list_arrow = ["up", "right", "down", "left"]
        self.animator.clear_elements()
        for i in range(4):
            position = list(zip(*np.where(policy == i)))
            self.animator.add_arrow(position, arrow[list_arrow[i]], "#04B153")

        self.animator.add_circle(list(zip(*np.where(policy == 4))))
        if not step:
            clear_output(wait=True)
        display(self.animator.fig)
