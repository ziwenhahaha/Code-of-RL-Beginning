# import gym
import gymnasium as gym
from matplotlib import pyplot as plt
from IPython import display
# Create CartPole environment
env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()

# Run the environment for 20 steps
for i in range(1000):
    # Display the current state of the environment
    env.render()
    
    # Choose a random action from the action space
    action = env.action_space.sample()
    
    # Take the chosen action and observe the next state, reward, and termination status
    state, reward, terminated, truncated, info = env.step(action)
    
    # If the episode is terminated or truncated, reset the environment
    if terminated or truncated:
        state, info = env.reset()

# Close the environment after exploration
env.close()
