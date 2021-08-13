from .heuristic_with_ml import HeuristicWithML
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np


class HeuristicWithDT(HeuristicWithML):
    def __init__(self, heuristic_func, use_model_only, is_continuous):
        super().__init__(heuristic_func, use_model_only)
        self.is_continuous = is_continuous
        self.model = DecisionTreeRegressor() if self.is_continuous else DecisionTreeClassifier()
        self.states = []
        self.actions = []

    def predict_action(self, observation, train):
        predicted_action = self.model.predict(np.array([observation]))[0]
        if self.is_continuous and not type(predicted_action) == np.ndarray:
            predicted_action = np.array([predicted_action])
        return predicted_action

    def store_transition(self, state, action, reward, state_, done):
        self.states.append(state)
        self.actions.append(action)

    def optimize(self, env, learning_type):
        self.model.fit(np.array(self.states), np.array(self.actions))