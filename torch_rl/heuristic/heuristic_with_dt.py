import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .heuristic_with_ml import HeuristicWithML


class HeuristicWithDT(HeuristicWithML):
    def __init__(self, heuristic_func, use_model_only, action_space, enable_action_blocking=False,
                 min_penalty=0, **args):
        super().__init__(heuristic_func, use_model_only, action_space, enable_action_blocking, min_penalty, **args)
        self.model = DecisionTreeRegressor() if self.is_continuous else DecisionTreeClassifier()
        self.states = []
        self.actions = []

    def predict_action(self, observation, train, **args):
        predicted_action = self.model.predict(np.array([observation]))[0]
        if self.is_continuous and not type(predicted_action) == np.ndarray:
            predicted_action = np.array([predicted_action])
        return predicted_action

    def store_transition(self, state, action, reward, state_, done):
        self.states.append(state)
        self.actions.append(action)

    def optimize(self, env, learning_type):
        self.model.fit(np.array(self.states), np.array(self.actions))
