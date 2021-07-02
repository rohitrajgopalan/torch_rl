import numpy as np
import torch as T
from .network import Network
from torch_rl.replay.replay import ReplayBuffer


class Agent:
    def __init__(self, is_double, gamma, epsilon, n_actions, input_dims,
                 mem_size, batch_size, fc_dims, optimizer_type, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, optimizer_args={}, randomized=False, goal=None):
        self.is_double = is_double
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.goal = goal
        if self.goal is not None:
            if not type(self.goal) == np.ndarray:
                self.goal = np.array([self.goal]).astype(np.float32)
            self.q_eval = Network(self.input_dims[0] + self.goal.shape[0], self.n_actions, fc_dims, optimizer_type,
                                  optimizer_args)
            self.q_next = Network(self.input_dims[0] + self.goal.shape[0], self.n_actions, fc_dims, optimizer_type,
                                  optimizer_args)
        else:
            self.q_eval = Network(self.input_dims[0], self.n_actions, fc_dims, optimizer_type, optimizer_args)
            self.q_next = Network(self.input_dims[0], self.n_actions, fc_dims, optimizer_type, optimizer_args)

        self.memory = ReplayBuffer(mem_size, input_dims, randomized=randomized, goal=self.goal)
        self.loss_history = []

    def choose_action(self, observation, train=True):
        if not train or np.random.random() > self.epsilon:
            if not type(observation) == np.ndarray:
                observation = np.array([observation]).astype(np.float32)
            state = T.tensor([observation], dtype=T.float32).to(self.q_eval.device)
            if self.goal is not None:
                goal = T.tensor([self.goal], dtype=T.float32).to(self.q_eval.device)
                inputs = T.cat([state, goal], dim=1)
            else:
                inputs = state
            _, advantage = self.q_eval.forward(inputs)
            return T.argmax(advantage).item()
        else:
            return np.random.choice(self.action_space)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done, goals = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        if goals is not None:
            goals = T.tensor(goals).to(self.q_eval.device)

        return states, actions, rewards, states_, dones, goals

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones, goals = self.sample_memory()

        if goals is not None:
            inputs = T.cat([states, goals], dim=1).float()
            inputs_ = T.cat([states_, goals], dim=1).float()
        else:
            inputs = states.float()
            inputs_ = states_.float()

        V_s, A_s = self.q_eval.forward(inputs)
        V_s_, A_s_ = self.q_next.forward(inputs_)

        indices = np.arange(self.batch_size)

        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        if self.is_double:
            V_s_eval, A_s_eval = self.q_eval.forward(inputs_)
            q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
            max_actions = T.argmax(q_eval, dim=1)
            q_next[dones] = 0.0
            q_target = rewards + self.gamma * q_next[indices, max_actions]
        else:
            q_next = q_next.max(dim=1)[0]
            q_next[dones] = 0.0
            q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

        self.loss_history.append(loss.item())
