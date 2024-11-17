import gymnasium as gym
import numpy as np  # 只需要下载numpy库即可

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
print(env.observation_space.shape[0])
# state = env.reset()[0]
# observation, reward, terminated, truncated, info = env.step(1)
# print(observation)
# print(type(reward))
# print(info)
# u_env = env.unwrapped
#
# for prob, next_state, reward, done in u_env.P[0][1]:
#     print(f"{prob=} {next_state=} {reward=} {done=}")
