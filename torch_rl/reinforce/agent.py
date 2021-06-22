import numpy as np
import torch as T
import torch.nn.functional as F
from .network import Network


class Agent:
    def __init__(self, gamma, n_actions, input_dims,
                 fc_dims, optimizer_type, optimizer_args={}, goal=None):
        self.gamma = gamma
        self.reward_memory = []
        self.log_prob_memory = []
        self.current_log_prob = 0.0
        self.loss_history = []

        self.goal = goal
        if self.goal is not None:
            if not type(self.goal) == np.ndarray:
                self.goal = np.array([self.goal]).astype(np.float32)
            self.policy = Network(input_dims[0] + self.goal.shape[0], n_actions, fc_dims, optimizer_type, optimizer_args)
        else:
            self.policy = Network(input_dims[0], n_actions, fc_dims, optimizer_type, optimizer_args)

    def choose_action(self, observation, train=True):
        if not type(observation) == np.ndarray:
            observation = np.array([observation]).astype(np.float32)
        state = T.tensor([observation], dtype=T.float32).to(self.policy.device)
        if self.goal is not None:
            goal = T.tensor([self.goal], dtype=T.float32).to(self.policy.device)
            inputs = T.cat([state, goal], dim=1)
        else:
            inputs = state
        probabilities = F.softmax(self.policy.forward(inputs), dim=1)
        if train:
            action_probs = T.distributions.Categorical(probabilities)
            action = action_probs.sample()
            log_probs = action_probs.log_prob(action)
            self.log_prob_memory.append(log_probs)

            return action.item()
        else:
            return T.argmax(probabilities).item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()

        # G_t = R_t+1 + gamma * R_t+2 + gamma**2 * R_t+3
        # G_t = sum from k=0 to k=T {gamma**k * R_t+k+1}
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = T.tensor(G, dtype=T.float).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G, self.log_prob_memory):
            loss += -g * logprob
        loss.backward()
        self.policy.optimizer.step()

        self.log_prob_memory = []
        self.reward_memory = []
        self.loss_history.append(loss.item())