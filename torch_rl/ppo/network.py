import torch as T
import torch.nn as nn
from torch.distributions import Normal

from torch_rl.utils.utils import get_torch_optimizer


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class Network(nn.Module):
    def __init__(self, input_dims, num_outputs, fc_dims, optimizer_type, optimizer_args={}, std=0.0):
        super(Network, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, num_outputs)
        )

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 1)
        )

        self.log_std = nn.Parameter(T.ones(1, num_outputs) * std)
        self.apply(init_weights)

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value
