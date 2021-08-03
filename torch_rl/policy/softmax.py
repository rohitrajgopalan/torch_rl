import torch as T
import torch.nn.functional as F
import numpy as np

from .policy import Policy


class SoftmaxPolicy(Policy):
    def __init__(self, num_actions, tau):
        super().__init__(num_actions)
        self.tau = tau

    def get_probs(self, **args):
        values = args['values']
        values = values/self.tau

        return F.softmax(T.tensor(values)).detach().numpy()

    def get_action(self, train, **args):
        if train:
            probs = self.get_probs(**args)
            return np.random.choice(self.num_actions, p=probs.squeeze())
        else:
            return np.random.choice(self.actions_with_max_value(args['values']))
