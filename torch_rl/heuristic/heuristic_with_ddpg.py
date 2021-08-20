import math

import torch as T
import torch.nn.functional as F

from torch_rl.ddpg.agent import Agent
from .heuristic_with_ml import HeuristicWithML


class HeuristicWithDDPG(HeuristicWithML, Agent):
    def __init__(self, heuristic_func, use_model_only, input_dims, action_space, tau, fc_dims, actor_optimizer_type,
                 critic_optimizer_type, actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
                 max_size=1000000, batch_size=64, goal=None, **args):
        HeuristicWithML.__init__(self, heuristic_func, use_model_only, action_space, False, 0, **args)
        Agent.__init__(input_dims, action_space, tau, fc_dims, actor_optimizer_type, critic_optimizer_type,
                       actor_optimizer_args, critic_optimizer_args, gamma,
                       max_size, batch_size, goal, False)

    def optimize(self, env, learning_type):
        num_updates = int(math.ceil(self.memory.mem_cntr / self.batch_size))
        for i in range(num_updates):
            start = (self.batch_size * (i - 1)) if i >= 1 else 0
            if i == num_updates - 1:
                end = self.memory.mem_cntr
            else:
                end = i * self.batch_size

            state, action, reward, new_state, done, goals = self.memory.sample_buffer(randomized=False, start=start, end=end)

            states = T.tensor(state)
            rewards = T.tensor(reward)
            dones = T.tensor(done)
            actions = T.tensor(action)
            states_ = T.tensor(new_state)

            if goals is not None:
                goals = T.tensor(goals).to(self.critic.device)
                inputs = T.cat([states, goals], dim=1).float()
                inputs_ = T.cat([states_, goals], dim=1).float()
            else:
                inputs = states.float()
                inputs_ = states_.float()

            target_actions = self.target_actor.forward(inputs_)
            target_actions = T.clamp(target_actions, self.action_space.low[0], self.action_space.high[0])
            critic_value_ = self.target_critic.forward(inputs_, target_actions)
            critic_value = self.critic.forward(inputs, actions)

            critic_value_[dones] = 0.0
            critic_value_ = critic_value_.view(-1)

            target = rewards + self.gamma * critic_value_
            target = target.view(self.batch_size, 1)

            self.critic.optimizer.zero_grad()
            critic_loss = F.mse_loss(target, critic_value)
            critic_loss.backward()
            self.critic.optimizer.step()

            self.actor.optimizer.zero_grad()
            actor_loss = self.critic.forward(states, self.actor.forward(states))
            actor_loss = -actor_loss.mean()
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_network_parameters()

    def predict_action(self, observation, train, **args):
        return Agent.choose_action(observation, args['t'], train)

    def store_transition(self, state, action, reward, state_, done):
        Agent.store_transition(self, state, action, reward, state_, done)
    
    