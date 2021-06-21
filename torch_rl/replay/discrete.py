import numpy as np


class DiscreteReplayBuffer:
    def __init__(self, max_size, input_shape, randomized=False, goal=None):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.randomized = randomized
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

        if goal is None:
            self.goal_memory = None
        else:
            self.goal_memory = np.full(shape=(self.mem_size, *input_shape), fill_value=goal, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        if self.randomized:
            batch = np.random.choice(max_mem, batch_size, replace=False)
        else:
            batch = np.arange(batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        if self.goal_memory is None:
            goals = None
        else:
            goals = self.goal_memory[batch]

        return states, actions, rewards, states_, terminal, goals
