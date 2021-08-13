import math

import numpy as np
import torch as T

from torch_rl.dueling_td.agent import DuelingTDAgent
from .heuristic_with_ml import HeuristicWithML
from ..utils.types import TDAlgorithmType, PolicyType, LearningType


class HeuristicWithDuelingTD(HeuristicWithML, DuelingTDAgent):
    def __init__(self, heuristic_func, use_model_only, algorithm_type, is_double, gamma, n_actions, input_dims,
                 mem_size, batch_size, fc_dims, optimizer_type, policy_type, policy_args={},
                 replace=1000, optimizer_args={}, goal=None):
        HeuristicWithML.__init__(self, heuristic_func, use_model_only)
        DuelingTDAgent.__init__(self, algorithm_type, is_double, gamma, n_actions, input_dims, mem_size, batch_size,
                                fc_dims,
                                optimizer_type, policy_type, policy_args, replace, optimizer_args, goal)

    def optimize(self, env, learning_type):
        num_updates = int(math.ceil(self.memory.mem_cntr / self.batch_size))
        for i in range(num_updates):
            self.q_next.optimizer.step()

            start = (self.batch_size * (i - 1)) if i >= 1 else 0
            if i == num_updates - 1:
                end = self.memory.mem_cntr
            else:
                end = i * self.batch_size

            state, action, reward, new_state, done, goals = self.memory.sample_buffer(randomized=False, start=start,
                                                                                      end=end)

            states = T.tensor(state)
            rewards = T.tensor(reward)
            dones = T.tensor(done)
            actions = T.tensor(action)
            states_ = T.tensor(new_state)

            indices = np.arange(end - start)
            if goals is not None:
                goals = T.tensor(goals).to(self.q_eval.device)
                inputs = T.cat([states, goals], dim=1).float()
                inputs_ = T.cat([states_, goals], dim=1).float()
            else:
                inputs = states.float()
                inputs_ = states_.float()

            V_s, A_s = self.q_eval.forward(inputs)
            V_s_, A_s_ = self.q_next.forward(inputs_)
            q_pred = T.add(V_s,
                           (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
            q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
            q_next[dones] = 0.0

            if self.is_double:
                V_s_eval, A_s_eval = self.q_eval.forward(states_)
                q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
                if self.algorithm_type == TDAlgorithmType.SARSA:
                    next_q_value = q_next[indices, self.determine_new_actions(env, learning_type, new_state, q_eval)]
                elif self.algorithm_type == TDAlgorithmType.Q:
                    next_q_value = q_next[indices, T.argmax(q_eval, dim=1)]
                elif self.algorithm_type == TDAlgorithmType.EXPECTED_SARSA:
                    q_eval[dones] = 0.0
                    next_q_value = self.get_weighted_sum(q_eval, new_state)
                else:
                    next_q_value = q_next[indices, actions]
            else:
                if self.algorithm_type == TDAlgorithmType.SARSA:
                    next_q_value = q_next[indices, self.determine_new_actions(env, learning_type, new_state)]
                elif self.algorithm_type == TDAlgorithmType.Q:
                    next_q_value = q_next[indices, T.argmax(q_next, dim=1)]
                elif self.algorithm_type == TDAlgorithmType.EXPECTED_SARSA:
                    next_q_value = self.get_weighted_sum(q_next, new_state)
                else:
                    next_q_value = q_next[indices, actions]
            q_target = rewards + self.gamma * next_q_value
            loss = self.q_eval.loss(q_target, q_pred)
            loss.backward()

            self.q_next.load_state_dict(self.q_eval.state_dict())

            if self.policy_type == PolicyType.EPSILON_GREEDY:
                self.policy.update()
            elif self.policy_type == PolicyType.UCB:
                for action, reward in zip(actions, rewards):
                    self.policy.update(action=action, reward=reward)

            self.q_eval.optimizer.zero_grad()
        self.memory.mem_cntr = 0

    def determine_new_actions(self, env, learning_type, next_states, q_values_arr=None):
        if learning_type == LearningType.OFFLINE:
            next_actions = np.zeros(next_states.shape[0], dtype=np.int64)
            for i, next_state in enumerate(next_states):
                next_actions[i] = self.heuristic_func(next_state)
        else:
            if q_values_arr is not None:
                q_values_arr = q_values_arr.detach().numpy()
                next_actions = np.zeros(q_values_arr.shape[0], dtype=np.int64)
                for i, q_values in enumerate(q_values_arr):
                    next_actions[i] = self.policy.get_action(True, values=q_values)
            else:
                next_actions = np.zeros(next_states.shape[0], dtype=np.int64)
                for i, next_state in enumerate(next_states):
                    next_actions[i] = self.get_action(env, LearningType.ONLINE, next_state, True)

        return T.tensor(next_actions)

    def predict_action(self, observation, train):
        return DuelingTDAgent.choose_action(observation, train)

    def store_transition(self, state, action, reward, state_, done):
        DuelingTDAgent.store_transition(self, state, action, reward, state_, done)
