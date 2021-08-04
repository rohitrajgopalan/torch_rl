import numpy as np
import math

from .policy import Policy


class UpperConfidenceBoundPolicy(Policy):
    def __init__(self, num_actions, confidence_factor):
        super().__init__(num_actions)
        self.confidence_factor = confidence_factor
        self.t = 0
        self.total_rewards = np.zeros((self.num_actions,))
        self.num_called = np.zeros((self.num_actions,))

    def update(self, **args):
        reward = args['reward']
        action = args['action']
        self.t += 1
        self.total_rewards[action] += reward
        self.num_called[action] += 1

    def get_action(self, train, **args):
        values = self.total_rewards / (self.num_called + 1)
        if train:
            values += (self.confidence_factor * np.sqrt(np.log(self.t+1) / (self.num_called + 1)))
        return np.random.choice(self.actions_with_max_value(values))

    def get_probs(self, **args):
        ucb_values = self.total_rewards / (self.num_called + 1)
        ucb_values += (self.confidence_factor * np.sqrt(np.log(self.t+1) / (self.num_called + 1)))

        actions_with_max_ucb_value = self.actions_with_max_value(ucb_values)
        policy_single = np.zeros((self.num_actions,))
        for a in actions_with_max_ucb_value:
            policy_single[a] = 1 / len(actions_with_max_ucb_value)

        values = args['values']
        policy_2d = np.zeros((values.shape[0], self.num_actions))
        for i in range(values.shape[0]):
            policy_2d[i] = policy_single

        return policy_2d

    def save_snapshot(self, file_name):
        np.save(file='{0}_total_rewards'.format(file_name), arr=self.total_rewards)
        np.save(file='{0}_num_called'.format(file_name), arr=self.num_called)

    def load_snapshot(self, file_name):
        self.total_rewards = np.load(file='{0}_total_rewards'.format(file_name))
        self.num_called = np.load(file='{0}_num_called'.format(file_name))
