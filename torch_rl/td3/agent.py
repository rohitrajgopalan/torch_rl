import torch as T
import torch.nn.functional as F
import numpy as np

from torch_rl.ddpg.network import PolicyNetwork, ValueNetwork
from torch_rl.noise.gaussian import GaussianExploration
from torch_rl.replay.continuous import ContinuousReplayBuffer


class Agent:
    def __init__(self, input_dims, action_space, tau, fc_dims, actor_optimizer_type, critic_optimizer_type,
                 actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
                 max_size=1000000, batch_size=64, randomized=False, policy_update_interval=2, noise_std=0.2,
                 noise_clip=0.5):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_update_interval = policy_update_interval
        self.noise_std = noise_std
        self.noise_clip = noise_clip

        self.memory = ContinuousReplayBuffer(max_size, input_dims, action_space.shape[0], randomized)

        self.noise = GaussianExploration(action_space)

        self.actor = PolicyNetwork(input_dims, action_space.shape[0], fc_dims, actor_optimizer_type,
                                   actor_optimizer_args)

        self.critic1 = ValueNetwork(input_dims, action_space.shape, fc_dims, critic_optimizer_type,
                                    critic_optimizer_args)

        self.critic2 = ValueNetwork(input_dims, action_space.shape, fc_dims, critic_optimizer_type,
                                    critic_optimizer_args)

        self.target_actor = PolicyNetwork(input_dims, action_space.shape[0], fc_dims, actor_optimizer_type,
                                          actor_optimizer_args)

        self.target_critic1 = ValueNetwork(input_dims, action_space.shape, fc_dims, critic_optimizer_type,
                                           critic_optimizer_args)
        self.target_critic2 = ValueNetwork(input_dims, action_space.shape, fc_dims, critic_optimizer_type,
                                           critic_optimizer_args)

        self.update_network_parameters(soft_tau=1)

        self.policy_loss_history = []
        self.value1_loss_history = []
        self.value2_loss_history = []
        self.learn_step_cntr = 0
        self.action_space = action_space

    def update_network_parameters(self, soft_tau=None):
        if soft_tau is None:
            soft_tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def choose_action(self, observation, t=0):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        action = self.actor.forward(state)
        action = action.detach().cpu().numpy()[0, 0]

        return self.noise.get_action(action, t)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)


    def sample_memory(self):
        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state, dtype=T.float).to(self.critic1.device)
        actions = T.tensor(action, dtype=T.float).to(self.critic1.device)
        rewards = T.tensor(reward, dtype=T.float).to(self.critic1.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.critic1.device)
        dones = T.tensor(done).to(self.actor.device)

        return states, actions, rewards, states_, dones

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.sample_memory()

        target_actions = self.target_actor.forward(states_)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=self.noise_clip)), -self.noise_clip,
                                                  self.noise_clip)
        target_actions = T.clamp(target_actions, self.action_space.low[0], self.action_space.high[0])

        critic_value1_ = self.target_critic1.forward(states_, target_actions)
        critic_value2_ = self.target_critic2.forward(states_, target_actions)
        critic_value1 = self.critic1.forward(states, actions)
        critic_value2 = self.critic2.forward(states, actions)

        critic_value1_[done] = 0.0
        critic_value2_[done] = 0.0

        critic_value1_ = critic_value1_.view(-1)
        critic_value2_ = critic_value2_.view(-1)

        critic_value_ = T.min(critic_value1_, critic_value2_)

        target = rewards + self.gamma*critic_value_
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

        self.policy_loss_history.append(abs(actor_loss.item()))
        self.value1_loss_history.append(critic_loss1.item())
        self.value2_loss_history.append(critic_loss2.item())
