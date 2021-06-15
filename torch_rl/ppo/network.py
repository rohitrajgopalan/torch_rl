import torch as T
import torch.nn as nn

from torch_rl.utils.utils import get_torch_optimizer, get_hidden_layer_sizes


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, fc_dims, optimizer_type, optimizer_args):
        super(PolicyNetwork, self).__init__()

        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)

        self.model = nn.Sequential(
            nn.Linear(*input_dim, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions)
        )

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.model(state)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, fc_dims, optimizer_type, optimizer_args):
        super(ValueNetwork, self).__init__()

        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)

        self.model = nn.Sequential(
            nn.Linear(*input_dim, fc1_dims),
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
