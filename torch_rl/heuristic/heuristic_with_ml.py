from torch_rl.utils.types import LearningType
import copy
import numpy as np


class HeuristicWithML:
    def __init__(self, heuristic_func, use_model_only):
        self.heuristic_func = heuristic_func
        self.use_model_only = use_model_only
        self.num_heuristic_actions_chosen = 0
        self.num_predicted_actions_chosen = 0

    def get_action(self, env, learning_type, observation, train=True):
        heuristic_action = self.heuristic_func(observation)
        predicted_action = self.predict_action(observation, train)
        if learning_type == LearningType.OFFLINE and train:
            self.num_heuristic_actions_chosen += 1
            return heuristic_action
        else:
            if self.use_model_only:
                self.num_predicted_actions_chosen += 1
                return predicted_action
            else:
                rewards = np.array([self.peek_reward(env, predicted_action), self.peek_reward(env, heuristic_action)])
                arg_max_reward = np.argmax(rewards)
                self.num_heuristic_actions_chosen += int(arg_max_reward == 1)
                self.num_predicted_actions_chosen += int(arg_max_reward == 0)
                return predicted_action if arg_max_reward == 0 else heuristic_action

    def peek_reward(self, env, action):
        local_env = copy.deepcopy(env)
        _, reward, _, _ = local_env.step(action)
        return reward

    def predict_action(self, observation, train):
        raise NotImplemented('Not implemented predict_action')

    def reset_metrics(self):
        self.num_heuristic_actions_chosen = 0
        self.num_predicted_actions_chosen = 0

    def store_transition(self, state, action, reward, state_, done):
        pass

    def optimize(self, env, learning_type):
        pass
