import random
import numpy as np
class ExperienceReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def getSize(self):
        return len(self.buffer)
        
    def add_expericence(self, experience):
        # state, action, reward, next_state, next_action, terminal
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        # batch = random.sample(self.buffer, batch_size, )
        batch = random.choices(self.buffer, k=batch_size)
        states, actions, rewards, next_states, next_actions, terminals = [],[],[],[],[],[]
        for experience in batch:
            state, action, reward, next_state, next_action, terminal = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            next_actions.append(next_action)
            terminals.append(terminal)
        return np.array(states),np.array(actions),np.array(rewards),np.array(next_states),np.array(next_actions),np.array(terminals)
        
    def sample_exps(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch