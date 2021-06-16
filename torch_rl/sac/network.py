import torch as T
import torch.nn as nn
import torch.nn.functional as F

from torch_rl.utils.utils import get_torch_optimizer, get_hidden_layer_sizes


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc_dims, n_actions, optimizer_type, optimizer_args):
        super(CriticNetwork, self).__init__()

        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)

        self.model = nn.Sequential(
            nn.Linear(input_dims[0] + n_actions, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        inputs = T.cat([state, action], dim=1)
        return self.model(inputs)


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc_dims, max_action,  n_actions, optimizer_type, optimizer_args):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action

        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)

        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.sigma = nn.Linear(fc2_dims, n_actions)

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True, train=True):
        if train:
            mu, sigma = self.forward(state)
            probabilities = T.distributions.Normal(mu, sigma)

            if reparameterize:
                actions = probabilities.rsample()  # reparameterizes the policy
            else:
                actions = probabilities.sample()

            action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
            log_probs = probabilities.log_prob(actions)
            log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
            log_probs = log_probs.sum(1, keepdim=True)

            return action, log_probs
        else:
            mu, _ = self.forward(state)
            action = T.tanh(mu) * T.tensor(self.max_action).to(self.device)
            return action, None


class ValueNetwork(nn.Module):
    def __init__(self, input_dims, fc_dims, optimizer_type, optimizer_args):
        super(ValueNetwork, self).__init__()

        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)

        self.model = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        return self.model(state)
