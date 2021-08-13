import numpy as np
import torch as T
from .network import DuelingTDNetwork
from torch_rl.replay.replay import ReplayBuffer
from torch_rl.utils.types import PolicyType, TDAlgorithmType
from torch_rl.utils.utils import choose_policy


class DuelingTDAgent:
    def __init__(self, algorithm_type, is_double, gamma, n_actions, input_dims,
                 mem_size, batch_size, fc_dims, optimizer_type, policy_type, policy_args={},
                 replace=1000, optimizer_args={}, goal=None):
        self.algorithm_type = algorithm_type
        self.is_double = is_double
        self.gamma = gamma
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.policy_type = policy_type
        self.policy = choose_policy(self.n_actions, self.policy_type, policy_args)

        self.replace_target_cnt = replace
        self.learn_step_counter = 0

        self.goal = goal
        if self.goal is not None:
            if not type(self.goal) == np.ndarray:
                self.goal = np.array([self.goal]).astype(np.float32)
            self.q_eval = DuelingTDNetwork(self.input_dims[0] + self.goal.shape[0], self.n_actions, fc_dims, optimizer_type,
                                           optimizer_args)
            self.q_next = DuelingTDNetwork(self.input_dims[0] + self.goal.shape[0], self.n_actions, fc_dims, optimizer_type,
                                           optimizer_args)
        else:
            self.q_eval = DuelingTDNetwork(self.input_dims[0], self.n_actions, fc_dims, optimizer_type, optimizer_args)
            self.q_next = DuelingTDNetwork(self.input_dims[0], self.n_actions, fc_dims, optimizer_type, optimizer_args)

        self.memory = ReplayBuffer(mem_size, input_dims, goal=self.goal)
        self.loss_history = []

    def choose_action(self, observation, train=True):
        if not type(observation) == np.ndarray:
            observation = np.array([observation]).astype(np.float32)
        state = T.tensor([observation], dtype=T.float32).to(self.q_eval.device)
        if self.goal is not None:
            goal = T.tensor([self.goal], dtype=T.float32).to(self.q_eval.device)
            inputs = T.cat([state, goal], dim=1)
        else:
            inputs = state
        _, advantage = self.q_eval.forward(inputs)
        advantages = advantage.cpu().detach().numpy().squeeze()
        return self.policy.get_action(train, values=advantages)

    def prepare_policy(self, q_values_arr, next_states):
        return self.policy.get_probs(values=q_values_arr, next_states=next_states)

    def get_weighted_sum(self, q_values_arr, next_states):
        q_values_arr = q_values_arr.detach().numpy()
        policy = self.prepare_policy(q_values_arr, next_states)
        return T.tensor(np.sum(policy * q_values_arr, axis=1, dtype=np.float32))

    def determine_actions_for_next_state_batch(self, next_states, q_values_arr=None):
        if q_values_arr is not None:
            q_values_arr = q_values_arr.detach().numpy()
            next_actions = np.zeros(q_values_arr.shape[0], dtype=np.int64)
            for i, q_values in enumerate(q_values_arr):
                next_actions[i] = self.policy.get_action(True, values=q_values)
        else:
            next_actions = np.zeros(next_states.shape[0], dtype=np.int64)
            for i, next_state in enumerate(next_states):
                next_actions[i] = self.choose_action(next_state)

        return T.tensor(next_actions)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done, goals = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        if goals is not None:
            goals = T.tensor(goals).to(self.q_eval.device)
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
        q_next[dones] = 0.0

        if self.is_double:
            V_s_eval, A_s_eval = self.q_eval.forward(states_)
            q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
            if self.algorithm_type == TDAlgorithmType.SARSA:
                next_q_value = q_next[indices, self.determine_actions_for_next_state_batch(new_state, q_eval)]
            elif self.algorithm_type == TDAlgorithmType.Q:
                next_q_value = q_next[indices, T.argmax(q_eval, dim=1)]
            elif self.algorithm_type == TDAlgorithmType.EXPECTED_SARSA:
                q_eval[dones] = 0.0
                next_q_value = self.get_weighted_sum(q_eval, new_state)
            else:
                next_q_value = q_next[indices, actions]
        else:
            if self.algorithm_type == TDAlgorithmType.SARSA:
                next_q_value = q_next[indices, self.determine_actions_for_next_state_batch(new_state)]
            elif self.algorithm_type == TDAlgorithmType.Q:
                next_q_value = q_next[indices, T.argmax(q_next, dim=1)]
            elif self.algorithm_type == TDAlgorithmType.EXPECTED_SARSA:
                next_q_value = self.get_weighted_sum(q_next, new_state)
            else:
                next_q_value = q_next[indices, actions]

        q_target = rewards + self.gamma * next_q_value
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        if self.policy_type == PolicyType.EPSILON_GREEDY:
            self.policy.update()
        elif self.policy_type == PolicyType.UCB:
            for reward, action in zip(rewards, actions):
                self.policy.update(reward=reward, action=action)
        elif self.policy_type == PolicyType.THOMPSON_SAMPLING:
            for reward in rewards:
                self.policy.update(reward=reward)

        self.loss_history.append(loss.item())

    def load_model(self, model_name):
        self.q_eval.load_model('{0}_q_eval'.format(model_name))
        self.q_next.load_model('{0}_q_next'.format(model_name))
        self.policy.load_snapshot(model_name)

    def save_model(self, model_name):
        self.q_eval.save_model('{0}_q_eval'.format(model_name))
        self.q_next.save_model('{0}_q_next'.format(model_name))
        self.policy.save_snapshot(model_name)