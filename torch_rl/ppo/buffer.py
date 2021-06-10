import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, randomized=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.randomized = randomized
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)

        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.log_probs_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.value_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.mask_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, log_prob, reward, value, mask):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.log_probs_memory[index] = log_prob
        self.reward_memory[index] = reward
        self.value_memory[index] = value
        self.mask_memory[index] = mask
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        if self.randomized:
            batch = np.random.choice(max_mem, batch_size, replace=False)
        else:
            batch = np.arange(batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        log_probs = self.log_probs_memory[batch]
        rewards = self.reward_memory[batch]
        values = self.value_memory[batch]
        masks = self.mask_memory[batch]

        return states, actions, log_probs, rewards, values, masks
