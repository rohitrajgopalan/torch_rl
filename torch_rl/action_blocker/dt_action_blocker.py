import numpy as np
from gym.spaces import Discrete

import joblib

from sklearn.tree import DecisionTreeClassifier

from torch_rl.replay.replay import ReplayBuffer


class DTActionBlocker:
    def __init__(self, action_space, **args):
        assert type(action_space) == Discrete

        self.action_space = action_space

        if 'model_name' in args and args['model_name'] is not None:
            self.model = joblib.load(args['model_name'])
        elif 'memory' in args and type(args['memory']) == ReplayBuffer:
            self.model = DecisionTreeClassifier()
            self.optimize(args['memory'], args['penalty'])

        self.num_actions_blocked = 0
        self.condensed_input_dims = 0

    def optimize(self, memory, penalty=0.01):
        states, actions, rewards, _, _ = memory.sample_buffer(randomized=False)
        if len(memory.input_shape) > 1:
            self.condensed_input_dims = 1
            for dim in memory.input_shape:
                self.condensed_input_dims *= dim
            state_input = np.zeros((min(memory.mem_cntr, memory.mem_size), self.condensed_input_dims))
            for i, state in enumerate(states):
                state_input[i] = state.flatten()
        else:
            state_input = states
            self.condensed_input_dims = memory.input_shape

        actions = actions.reshape(-1, 1)

        inputs = np.concatenate((state_input, actions), axis=1)
        outputs = (rewards <= -penalty).astype(int)

        self.model.fit(inputs, outputs)

    def block_action(self, env, state, action):
        if len(state.shape) > 1:
            state = state.flatten()
        state = state.reshape(-1, self.condensed_input_dims)
        action = np.array([action]).reshape(-1, 1)

        input_arr = np.concatenate((state, action), axis=1)
        output = self.model.predict(np.array([input_arr]))[0]

        return output == 1

    def find_safe_action(self, env, state, initial_action,):
        if self.block_action(env, state, initial_action):
            remaining_actions = [a for a in range(self.action_space.n) if a != initial_action]
            for a in remaining_actions:
                if not self.block_action(env, state, a):
                    return a
                self.num_actions_blocked += 1
            return None
        else:
            return initial_action
