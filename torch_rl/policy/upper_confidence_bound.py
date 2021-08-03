import numpy as np

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
        values = self.total_rewards / self.num_called
        if train:
            values += (self.confidence_factor * np.sqrt(np.log(self.t) / self.num_called))
        return np.random.choice(self.actions_with_max_value(values))

    def get_probs(self, **args):
        ucb_values = self.total_rewards / self.num_called
        ucb_values += (self.confidence_factor * np.sqrt(np.log(self.t) / self.num_called))

        actions_with_max_ucb_value = self.actions_with_max_value(ucb_values)
        policy_single = np.zeros((self.num_actions,))
        for a in actions_with_max_ucb_value:
            policy_single[a] = 1 / len(actions_with_max_ucb_value)

        values = args['values']
        policy_2d = np.zeros((values.shape[0], self.num_actions))
        for i in range(values.shape[0]):
            policy_2d[i] = policy_single

        return policy_2d
