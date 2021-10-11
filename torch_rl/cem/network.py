import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_dims, action_dims, network_args):
        super(Network, self).__init__()

        self.input_dim_len = len(input_dims)
        self.total_fc_input_dims = 1
        for dim in input_dims:
            self.total_fc_input_dims *= dim

        self.h_size = network_args['fc_dim']
        self.a_size = action_dims[0]

        self.fc1 = nn.Linear(self.total_fc_input_dims, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def set_weights(self, weights):
        s_size = self.total_fc_input_dims
        h_size = self.h_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size * h_size) + h_size
        fc1_W = T.from_numpy(weights[:s_size * h_size].reshape(s_size, h_size))
        fc1_b = T.from_numpy(weights[s_size * h_size:fc1_end])
        fc2_W = T.from_numpy(weights[fc1_end:fc1_end + (h_size * a_size)].reshape(h_size, a_size))
        fc2_b = T.from_numpy(weights[fc1_end + (h_size * a_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def get_weights_dim(self):
        return (self.s_size + 1) * self.h_size + (self.h_size + 1) * self.a_size

    def forward(self, state):
        if len(state.shape) == self.input_dim_len:
            state = state.flatten()
            x = F.relu(self.fc1(state))
        else:
            states = T.empty((state.shape[0], self.input_dim_len))
            for i, s in enumerate(state):
                states[i] = s.flatten()
            x = F.relu(self.fc1(states))

        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x.cpu().data

    def save_model(self, model_file_name):
        T.save(self.state_dict(), model_file_name)

    def load_model(self, model_file_name):
        self.load_state_dict(T.load(model_file_name))
