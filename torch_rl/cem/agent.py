import math
import torch as T
import numpy as np
from .network import Network
from ..replay.replay import ReplayBuffer


class CEMAgent:
    def __init__(self, input_dims, action_space, network_args, gamma=1.0, pop_size=50, elite_frac=0.2, sigma=0.5,
                 goal=None):
        self.gamma = gamma

        self.goal = goal
        if self.goal is not None:
            if not type(self.goal) == np.ndarray:
                self.goal = np.array([self.goal]).astype(np.float32)

            self.network = Network(tuple(np.add(input_dims, self.goal.shape)), action_space.shape[0], network_args)
        else:
            self.network = Network(input_dims, action_space.shape[0], network_args)

        self.pop_size = pop_size
        self.n_elite = int(pop_size * elite_frac)
        self.sigma = sigma

        self.best_weights = sigma * np.random.randn(self.network.get_weights_dim())

    def choose_action(self, observation, train=True):
        if not train:
            self.network.set_weights(self.best_weights)

        state = T.tensor(observation, dtype=T.float).to(self.network.device)
        if self.goal is not None:
            goal = T.tensor(self.goal, dtype=T.float).to(self.network.device)
            inputs = T.cat([state, goal], dim=0)
        else:
            inputs = state
        action = self.network.forward(inputs)
        return action.cpu().detach().numpy()

    def evaluate_weights(self, env, weights):
        self.network.set_weights(weights)
        episode_return = 0.0
        state = env.reset()
        done = False
        t = 0
        while not done:
            action = self.choose_action(state, True)
            state, reward, done, _ = env.step(action)
            episode_return += reward * math.pow(self.gamma, t)
            t += 1

        return episode_return

    def learn(self, env):
        weights_pop = [self.best_weights + (self.sigma * np.random.randn(self.network.get_weights_dim()))
                       for _ in range(self.pop_size)]
        rewards = np.array([self.evaluate_weights(env, weights) for weights in weights_pop])
        elite_idxs = rewards.argsort()[-self.n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        self.best_weights = np.array(elite_weights).mean(axis=0)

