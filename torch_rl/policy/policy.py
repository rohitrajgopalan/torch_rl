import numpy as np


class Policy:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def actions_with_max_value(self, values):
        actions_with_max_value = []

        max_value = np.max(values)

        for a in range(values.shape[0]):
            if values[a] == max_value:
                actions_with_max_value.append(a)

        return actions_with_max_value

    def get_action(self, train, **args):
        pass

    def update(self, **args):
        pass

    def get_probs(self, **args):
        pass

    def save_snapshot(self, file_name):
        pass

    def load_snapshot(self, file_name):
        pass
