import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_action_dims=1, randomized=False, goal=None):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.randomized = randomized
        self.state_memory = np.zeros((self.mem_size, *input_shape)).astype(np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape)).astype(np.float32)

        if n_action_dims == 1:
            self.action_memory = np.zeros(self.mem_size)
        else:
            self.action_memory = np.zeros((self.mem_size, n_action_dims))

        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        if goal is not None:
            self.goal_memory = np.full((self.mem_size, goal.shape[0]), goal).astype(np.float32)
        else:
            self.goal_memory = None

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        if self.randomized:
            batch = np.random.choice(max_mem, batch_size)
        else:
            batch = np.arange(batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        if self.goal_memory is None:
            goals = None
        else:
            goals = self.goal_memory[batch]

        return states, actions, rewards, states_, dones, goals
