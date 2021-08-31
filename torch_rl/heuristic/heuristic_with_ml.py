from gym.spaces import Discrete, Box

from torch_rl.action_blocker.action_blocker import ActionBlocker
from torch_rl.utils.types import LearningType
import copy
import numpy as np


class HeuristicWithML:
    def __init__(self, heuristic_func, use_model_only, action_space, enable_action_blocking=False, min_penalty=0,
                 **args):
        self.heuristic_func = heuristic_func
        self.use_model_only = use_model_only
        self.is_continuous = type(action_space) == Box
        self.num_heuristic_actions_chosen = 0
        self.num_predicted_actions_chosen = 0
        if type(action_space) == Discrete and enable_action_blocking:
            self.enable_action_blocking = True
            self.action_blocker = ActionBlocker(action_space, min_penalty)
        else:
            self.enable_action_blocking = False
            self.action_blocker = None
        self.initial_action_blocked = False
        self.initial_action = None
        
        for arg in args:
            setattr(self, arg, args[arg])

    def get_action(self, env, learning_type, observation, train=True, **args):
        self.initial_action = self.get_original_action(learning_type, observation, train, **args)
        if self.enable_action_blocking:
            actual_action = self.action_blocker.find_safe_action(env, observation, self.initial_action)
            self.initial_action_blocked = (actual_action is None or actual_action != self.initial_action)
            if actual_action is None:
                print('WARNING: No valid policy action found, running original action')
            return self.initial_action if actual_action is None else actual_action
        else:
            return self.initial_action

    def get_original_action(self, learning_type, observation, train=True, **args):
        heuristic_action = self.heuristic_func(self, observation)
        predicted_action = self.predict_action(observation, train, **args)
        if learning_type == LearningType.OFFLINE and train:
            self.num_heuristic_actions_chosen += 1
            return heuristic_action
        else:
            if self.use_model_only:
                self.num_predicted_actions_chosen += 1
                return predicted_action
            else:
                if type(predicted_action) == np.ndarray and type(heuristic_action) == np.ndarray:
                    if np.array_equal(predicted_action, heuristic_action):
                        self.num_predicted_actions_chosen += 1
                        return predicted_action
                    else:
                        self.num_heuristic_actions_chosen += 1
                        return heuristic_action
                else:
                    if predicted_action == heuristic_action:
                        self.num_predicted_actions_chosen += 1
                        return predicted_action
                    else:
                        self.num_heuristic_actions_chosen += 1
                        return heuristic_action

    def predict_action(self, observation, train, **args):
        raise NotImplemented('Not implemented predict_action')

    def reset_metrics(self):
        self.num_heuristic_actions_chosen = 0
        self.num_predicted_actions_chosen = 0

    def store_transition(self, state, action, reward, state_, done):
        pass

    def optimize(self, env, learning_type):
        pass
