import numpy as np
import torch as T
from gym.spaces import Discrete
from .network import Network


class ActionBlocker:
    def __init__(self, input_dims, action_space, fc_dims, optimizer_type, optimizer_args, randomized_replay=False,
                 mem_size=1000, min_penalty=0, batch_size=64):
        assert min_penalty > 0
        self.min_penalty = min_penalty
        self.batch_size = batch_size
        self.action_dim = 1 if type(action_space) == Discrete else action_space.shape[0]
        self.network = Network(input_dims, self.action_dim, fc_dims, optimizer_type, optimizer_args)
        self.randomized_replay = randomized_replay
        self.confusion_matrix = {'t_p': 0, 'f_p': 0, 't_n': 0, 'f_n': 0}
        self.mem_size = mem_size
        self.mem_cntr = 0
        if self.action_dim == 1:
            self.actions = np.zeros(self.mem_size, dtype=np.int32)
        else:
            self.actions = np.zeros((self.mem_size, self.action_dim), dtype=np.float32)
        self.states = np.zeros((self.mem_size, input_dims[0]))
        self.actual_logits = np.zeros(self.mem_size)
        self.num_actions_blocked = 0

    def should_action_be_blocked(self, state, action):
        if self.action_dim == 1:
            action = T.tensor([action], dtype=T.int32)
        else:
            action = T.tensor(action, dtype=T.float32)

        state = T.tensor(state, dtype=T.float32)

        results = T.round(T.sigmoid(self.network.forward(state, action)))
        result = results.cpu().detach().numpy()[0]

        return result == 1

    def update_confusion_metrix(self, state, action, pred_logit, reward):
        index = self.mem_cntr % self.mem_size

        self.num_actions_blocked += 1 if pred_logit == 1 else 0

        actual_logit = 1 if reward <= -self.min_penalty else 0
        self.states[index] = state
        self.actions[index] = action
        self.actual_logits[index] = actual_logit

        if pred_logit != actual_logit:
            if pred_logit == 0:
                self.confusion_matrix['f_n'] += 1
            else:
                self.confusion_matrix['f_p'] += 1
        else:
            if pred_logit == 0:
                self.confusion_matrix['t_n'] += 1
            else:
                self.confusion_matrix['t_p'] += 1

        self.mem_cntr += 1

    def learn(self):
        if self.mem_cntr < self.batch_size:
            pass

        max_mem = min(self.mem_cntr, self.mem_size)

        if self.randomized_replay:
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
        else:
            batch = np.arange(self.batch_size)

        states = self.states[batch]
        actions = self.actions[batch]
        y_true = self.actual_logits[batch]

        states = T.tensor(states).to(self.network.device)
        actions = T.tensor(actions).to(self.network.device)
        y_true = T.tensor(y_true).to(self.network.device)

        self.network.optimizer.zero_grad()
        y_pred = T.round(T.sigmoid(self.network.forward(states, actions)))
        loss = self.network.loss(y_true, y_pred).to(self.network.device)
        loss.backward()
        self.network.optimizer.step()

    def reset_confusion_matrix(self):
        self.confusion_matrix = {'t_p': 0, 'f_p': 0, 't_n': 0, 'f_n': 0}
        self.num_actions_blocked = 0

    def get_precision_and_recall(self):
        true_positives = self.confusion_matrix['t_p']
        false_positives = self.confusion_matrix['f_p']
        false_negatives = self.confusion_matrix['f_n']

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        return precision, recall
