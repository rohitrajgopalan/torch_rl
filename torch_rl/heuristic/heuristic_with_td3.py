import math

import numpy as np
import torch as T
import torch.nn.functional as F

from torch_rl.td3.agent import TD3Agent
from .heuristic_with_ml import HeuristicWithML


class HeuristicWithTD3(HeuristicWithML, TD3Agent):
    def __init__(self, heuristic_func, use_model_only, input_dims, action_space, tau, network_args,
                 actor_optimizer_type,
                 critic_optimizer_type, actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
                 max_size=1000000, batch_size=64, policy_update_interval=2, noise_std=0.2,
                 noise_clip=0.5, goal=None, model_name=None, **args):
        HeuristicWithML.__init__(self, heuristic_func, use_model_only, action_space, False, 0, **args)
        TD3Agent.__init__(input_dims, action_space, tau, network_args, actor_optimizer_type, critic_optimizer_type,
                          actor_optimizer_args, critic_optimizer_args, gamma,
                          max_size, batch_size, policy_update_interval, noise_std,
                          noise_clip, goal, False, model_name)

    def optimize(self, env, learning_type):
        num_updates = int(math.ceil(self.memory.mem_cntr / self.batch_size))
        for i in range(num_updates):
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

            if goals is not None:
                goals = T.tensor(goals).to(self.actor.device)
                inputs = T.cat([states, goals], dim=1).float()
                inputs_ = T.cat([states_, goals], dim=1).float()
            else:
                inputs = states.float()
                inputs_ = states_.float()

            target_actions = self.target_actor.forward(inputs_)
            target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=self.noise_clip)),
                                                      -self.noise_clip,
                                                      self.noise_clip)
            target_actions = T.clamp(target_actions, self.action_space.low[0], self.action_space.high[0])

            critic_value1_ = self.target_critic1.forward(inputs_, target_actions)
            critic_value2_ = self.target_critic2.forward(inputs_, target_actions)
            critic_value1 = self.critic1.forward(inputs, actions)
            critic_value2 = self.critic2.forward(inputs, actions)

            critic_value1_[dones] = 0.0
            critic_value2_[dones] = 0.0

            critic_value1_ = critic_value1_.view(-1)
            critic_value2_ = critic_value2_.view(-1)

            critic_value_ = T.min(critic_value1_, critic_value2_)

            target = rewards + self.gamma * critic_value_
            target = target.view(self.batch_size, 1)

            self.critic1.optimizer.zero_grad()
            self.critic2.optimizer.zero_grad()
            critic_loss1 = F.mse_loss(target, critic_value1)
            critic_loss2 = F.mse_loss(target, critic_value2)
            critic_loss = critic_loss1 + critic_loss2
            critic_loss.backward()

            self.critic1.optimizer.step()
            self.critic2.optimizer.step()

            self.learn_step_cntr += 1

            if self.learn_step_cntr % self.policy_update_interval != 0:
                return

            self.actor.optimizer.zero_grad()
            actor_loss1 = self.critic1.forward(states, self.actor.forward(states))
            actor_loss = -actor_loss1.mean()
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_network_parameters()

    def predict_action(self, observation, train, **args):
        return TD3Agent.choose_action(observation, args['t'], train)

    def store_transition(self, state, action, reward, state_, done):
        TD3Agent.store_transition(self, state, action, reward, state_, done)

    def __str__(self):
        return 'Heuristic driven TD3 Agent {0}'.format('only using models' if self.use_model_only else 'alternating '
                                                                                                       'between '
                                                                                                       'models and '
                                                                                                       'heuristic')
