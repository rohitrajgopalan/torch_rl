import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch_rl.utils.utils import get_torch_optimizer, get_hidden_layer_sizes


class PolicyNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, network_args, optimizer_type, optimizer_args, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.input_dim_len = len(input_dims)
        self.conv_list = []
        if 'cnn_dims' in network_args:
            cnn_dims = network_args['cnn_dims']
            for i, cnn_dim in enumerate(cnn_dims):
                in_channel = input_dims[0] if i == 0 else cnn_dims[i - 1][0]
                out_channel = cnn_dim[0]
                kernel_size = cnn_dim[1]
                stride = cnn_dim[2]
                self.conv_list.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride))

                self.total_fc_input_dims = self.calculate_conv_output_dims(input_dims)
            else:
                self.total_fc_input_dims = 1
                for dim in input_dims:
                    self.total_fc_input_dims *= dim

        fc_dims = network_args['fc_dims']
        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)
        self.fc1 = nn.Linear(self.total_fc_input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        out = T.zeros(1, *input_dims)
        for conv in self.conv_list:
            out = conv(out)
        return int(np.prod(out.size()))

    def forward(self, state):
        if len(self.conv_list) > 0:
            out = state
            for conv in self.conv_list:
                out = F.relu(conv(out))
            conv3 = out
            state = conv3.view(conv3.size()[0], -1)
            x = F.relu(self.fc1(state))
        else:
            if len(state.shape) == self.input_dim_len:
                state = state.flatten()
                x = F.relu(self.fc1(state))
            else:
                states = T.empty((state.shape[0], self.input_dim_len))
                for i, s in enumerate(state):
                    states[i] = s.flatten()
                x = F.relu(self.fc1(states))

        x = F.relu(self.fc2(x))
        return T.tanh(self.fc3(x))

    def save_model(self, model_file_name):
        T.save(self.state_dict(), model_file_name)

    def load_model(self, model_file_name):
        self.load_state_dict(T.load(model_file_name))


class ValueNetwork(nn.Module):
    def __init__(self, input_dims, action_dim, network_args, optimizer_type, optimizer_args, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.conv_list = []
        self.input_dim_len = len(input_dims)
        if 'cnn_dims' in network_args:
            cnn_dims = network_args['cnn_dims']
            for i, cnn_dim in enumerate(cnn_dims):
                in_channel = input_dims[0] if i == 0 else cnn_dims[i - 1][0]
                out_channel = cnn_dim[0]
                kernel_size = cnn_dim[1]
                stride = cnn_dim[2]
                self.conv_list.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride))

            self.total_fc_input_dims = self.calculate_conv_output_dims(tuple(np.add(input_dims, action_dim)))
        else:
            self.total_fc_input_dims = 1
            for dim in input_dims:
                self.total_fc_input_dims *= dim
            self.total_fc_input_dims += action_dim[0]

        fc_dims = network_args['fc_dims']
        fc1_dims, fc2_dims = get_hidden_layer_sizes(fc_dims)

        self.fc1 = nn.Linear(self.total_fc_input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)

        self.optimizer = get_torch_optimizer(self.parameters(), optimizer_type, optimizer_args)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        out = T.zeros(1, *input_dims)
        for conv in self.conv_list:
            out = conv(out)
        return int(np.prod(out.size()))

    def forward(self, state, action):
        if len(self.conv_list) > 0:
            out = T.cat([state, action], 1)
            for conv in self.conv_list:
                out = F.relu(conv(out))
            conv3 = out
            x = conv3.view(conv3.size()[0], -1)
        else:
            if len(state.shape) == self.input_dim_len:
                state = state.flatten()
                x = T.cat([state, action], 1)
            else:
                states = T.empty((state.shape[0], self.input_dim_len))
                for i, s in enumerate(state):
                    states[i] = s.flatten()
                x = T.cat([states, action], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_model(self, model_file_name):
        T.save(self.state_dict(), model_file_name)

    def load_model(self, model_file_name):
        self.load_state_dict(T.load(model_file_name))
