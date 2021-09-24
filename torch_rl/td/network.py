import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch_rl.utils.utils import get_torch_optimizer, get_hidden_layer_sizes


class FCTDNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc_dims, optimizer_type, optimizer_args={}, use_mse=True):
        super(FCTDNetwork, self).__init__()

        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)

        self.input_dim_len = len(input_dims)
        self.total_fc_dims = 1
        for dim in input_dims:
            self.total_fc_dims *= dim

        self.model = nn.Sequential(
            nn.Linear(self.total_fc_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions)
        )

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)
        self.loss = nn.MSELoss() if use_mse else nn.SmoothL1Loss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        if len(state.shape) == self.input_dim_len:
            state = state.flatten()
            return self.model(state)
        else:
            states = T.empty((state.shape[0], self.input_dim_len))
            for i, s in enumerate(state):
                states[i] = s.flatten()
            return self.model(states)

    def save_model(self, model_file_name):
        T.save(self.state_dict(), model_file_name)

    def load_model(self, model_file_name):
        self.load_state_dict(T.load(model_file_name))


class CNNTDNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, cnn_dims, fc_dim, optimizer_type, optimizer_args={}, use_mse=True):
        super(CNNTDNetwork, self).__init__()
        assert type(cnn_dims) == list

        self.conv_list = []

        for i, cnn_dim in enumerate(cnn_dims):
            in_channel = input_dims[0] if i == 0 else cnn_dims[i-1][0]
            out_channel = cnn_dim[0]
            kernel_size = cnn_dim[1]
            stride = cnn_dim[2]
            self.conv_list.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride))

        total_fc_input_dims = self.calculate_conv_output_dims(input_dims)
        self.fc1 = nn.Linear(total_fc_input_dims, fc_dim)
        self.fc2 = nn.Linear(fc_dim, n_actions)

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)
        self.loss = nn.MSELoss() if use_mse else nn.SmoothL1Loss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        out = T.zeros(1, *input_dims)
        for conv in self.conv_list:
            out = conv(out)
        return int(np.prod(out.size()))

    def forward(self, state):
        out = state
        for conv in self.conv_list:
            out = F.relu(conv(out))
        conv3 = out
        conv_state = conv3.view(conv3.size()[0], -1)

        flat1 = F.relu(self.fc1(conv_state))
        return self.fc2(flat1)

    def save_model(self, model_file_name):
        T.save(self.state_dict(), model_file_name)

    def load_model(self, model_file_name):
        self.load_state_dict(T.load(model_file_name))


def choose_network(input_dims, n_actions, network_args, optimizer_type, optimizer_args={}, use_mse=True):
    if 'cnn_dims' and 'fc_dim' in network_args:
        return CNNTDNetwork(input_dims, n_actions, network_args['cnn_dims'], network_args['fc_dim'],
                            optimizer_type, optimizer_args, use_mse)
    elif 'fc_dims' in network_args:
        return FCTDNetwork(input_dims, n_actions, network_args['fc_dims'], optimizer_type, optimizer_args, use_mse)
    else:
        return TypeError('Invalid arguments to choose Network type')