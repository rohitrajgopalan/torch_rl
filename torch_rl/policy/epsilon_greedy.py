import numpy as np
from .policy import Policy


class EpsilonGreedyPolicy(Policy):
    def __init__(self, num_actions, enable_decay, eps_start, eps_min, eps_dec):
        super().__init__(num_actions)
        self.enable_decay = enable_decay
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_dec = eps_dec

    def get_action(self, train, **args):
        if not train or np.random.random() > self.epsilon:
            return np.random.choice(self.actions_with_max_value(args['values']))
        else:
            return np.random.choice(self.num_actions)

    def update(self, **args):
        if self.enable_decay:
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def get_probs(self, **args):
        values = args['values']
        policy = np.zeros((values.shape[0], self.num_actions))
        for i, q_values in enumerate(values):
            actions_with_max_value = self.actions_with_max_value(q_values)
            for a in range(self.num_actions):
                if a in actions_with_max_value:
                    policy[i, a] = (1 - self.epsilon) / len(actions_with_max_value)
                else:
                    policy[i, a] = self.epsilon / (self.num_actions - len(actions_with_max_value))
        return policy
