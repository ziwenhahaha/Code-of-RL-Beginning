import gymnasium as gym
import torch


desc = [
    "FFFFF",
    "FHHFF",
    "FFHFF",
    "FHGHS",
    "FHFFF",
]

text_env = gym.make(
    "FrozenLake-v1",
    desc=desc,
    is_slippery=False,
    render_mode="human",
)
state = text_env.reset()[0]

done = False
while True:
    action = 0
    state, reward, terminated, truncated, info = text_env.step(action)
    print(f"{state=},{action=}")
    text_env.render()  # 渲染环境
    done = terminated or truncated

text_env.close()
#
