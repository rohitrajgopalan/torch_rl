import copy

from gym.spaces import Discrete


class ActionBlocker:
    def __init__(self, action_space, min_penalty=0):
        assert min_penalty > 0
        assert type(action_space) == Discrete
        self.min_penalty = min_penalty
        self.num_actions_blocked = 0
        self.action_space = action_space

    def block_action(self, env, action):
        local_env = copy.deepcopy(env)
        _, reward, _, _ = local_env.step(action)
        return reward <= -self.min_penalty

    def find_safe_action(self, env, initial_action):
        if self.block_action(env, initial_action):
            remaining_actions = [a for a in range(self.action_space.n) if a != initial_action]
            for a in remaining_actions:
                if not self.block_action(env, a):
                    return a
                self.num_actions_blocked += 1
            return None
        else:
            return initial_action
