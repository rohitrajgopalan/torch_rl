import torch as T
import torch.nn as nn
from torch_rl.utils.utils import get_torch_optimizer


class Network(nn.Module):
    def __init__(self, input_dims, n_actions, fc_dims, optimizer_type, optimizer_args={}):
        super(Network, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(*input_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions)
        )

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.model(state)
