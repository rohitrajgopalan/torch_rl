import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch_rl.utils.utils import get_torch_optimizer, get_hidden_layer_sizes


class DuelingTDNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, network_args, optimizer_type, optimizer_args={}, use_mse=True):
        super(DuelingTDNetwork, self).__init__()

        self.input_dim_len = len(input_dims)
        self.total_fc_input_dims = 1
        for dim in input_dims:
            self.total_fc_input_dims *= dim

        fc_dims = network_args['fc_dims']
        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)

        self.fc1 = nn.Linear(self.total_fc_input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions)

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)
        self.loss = nn.MSELoss() if use_mse else nn.SmoothL1Loss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        if len(state.shape) == self.input_dim_len:
            state = state.flatten()
            flat1 = F.relu(self.fc1(state))
        else:
            states = T.empty((state.shape[0], self.input_dim_len))
            for i, s in enumerate(state):
                states[i] = s.flatten()
            flat1 = F.relu(self.fc1(states))

        flat2 = F.relu(self.fc2(flat1))

        V = self.V(flat2)
        A = self.A(flat2)

        return V, A

    def save_model(self, model_file_name):
        T.save(self.state_dict(), model_file_name)

    def load_model(self, model_file_name):
        self.load_state_dict(T.load(model_file_name))
