import random
import numpy as np

from .policy import Policy


class ThompsonSamplingPolicy(Policy):
    def __init__(self, num_actions, min_penalty):
        super().__init__(num_actions)
        self.min_penalty = min_penalty
        self.successes = np.zeros((self.num_actions,))
        self.failures = np.zeros((self.num_actions,))

    def get_action(self, train, **args):
        if train:
            beta_values = np.array([random.betavariate(self.successes[a] + 1, self.failures[a] + 1)
                                    for a in range(self.num_actions)])
            return np.random.choice(self.actions_with_max_value(beta_values))
        else:
            return np.random.choice(self.actions_with_max_value(self.successes))

    def update(self, **args):
        reward = args['reward']
        action = args['action']
        if reward <= -self.min_penalty:
            self.failures[action] += 1
        else:
            self.successes[action] += 1

    def get_probs(self, **args):
        beta_values = np.array([random.betavariate(self.successes[a] + 1, self.failures[a] + 1)
                                for a in range(self.num_actions)])
        actions_with_max_beta_value = self.actions_with_max_value(beta_values)
        policy_single = np.zeros((self.num_actions,))
        for a in actions_with_max_beta_value:
            policy_single[a] = 1 / len(actions_with_max_beta_value)

        values = args['values']
        policy_2d = np.zeros((values.shape[0], self.num_actions))
        for i in range(values.shape[0]):
            policy_2d[i] = policy_single

        return policy_2d
