import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch_rl.utils.utils import get_torch_optimizer


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, fc_dims, optimizer_type, optimizer_args, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dim, fc_dims)
        self.fc2 = nn.Linear(fc_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, n_actions)

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, fc_dims, optimizer_type, optimizer_args, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dim + action_dim, fc_dims)
        self.fc2 = nn.Linear(fc_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, 1)

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
