import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch_rl.utils.utils import get_torch_optimizer, get_hidden_layer_sizes


class Network(nn.Module):
    def __init__(self, input_dims, n_actions, fc_dims, optimizer_type, optimizer_args={}):
        super(Network, self).__init__()

        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return pi, v
