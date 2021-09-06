from .heuristic_with_ml import HeuristicWithML
from torch_rl.hill_climbing.agent import HillClimbingAgent


class HeuristicWithHillClimbing(HeuristicWithML, HillClimbingAgent):
    def __init__(self, heuristic_func, use_model_only, input_dims, action_space, gamma, noise_scale=1e-2,
                 enable_action_blocking=False, min_penalty=0,
                 use_ml_for_action_blocker=False, action_blocker_memory=None, action_blocker_model_name=None,
                 **args):
        HeuristicWithML.__init__(self, heuristic_func, use_model_only, action_space, enable_action_blocking,
                                 min_penalty, use_ml_for_action_blocker, action_blocker_memory,
                                 action_blocker_model_name, **args)
        HillClimbingAgent.__init__(self, input_dims, action_space, gamma, noise_scale, enable_action_blocking,
                                   min_penalty, use_ml_for_action_blocker, action_blocker_memory,
                                   action_blocker_memory)

    def optimize(self, env, learning_type):
        self.learn()

    def predict_action(self, observation, train, **args):
        return HillClimbingAgent.choose_policy_action(observation, train)

    def store_transition(self, state, action, reward, state_, done):
        HillClimbingAgent.store_transition(self, state, action, reward, state_, done)



