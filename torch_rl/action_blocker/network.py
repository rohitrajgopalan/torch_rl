import torch as T
import torch.nn as nn

from torch_rl.utils.utils import get_torch_optimizer, get_hidden_layer_sizes


class Network(nn.Module):
    def __init__(self, input_dim, action_dim, fc_dims, optimizer_type, optimizer_args):
        super(Network, self).__init__()

        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)

        self.model = nn.Sequential(
            nn.Linear(input_dim[0] + action_dim[0], fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
            nn.Sigmoid()
        )

        self.loss = nn.BCELoss()
        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        inputs = T.cat([state, action], dim=1)
        return self.model(inputs)

