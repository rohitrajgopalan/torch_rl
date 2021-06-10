import numpy as np
import torch as T
import torch.nn.functional as F
from .network import PolicyNetwork, ValueNetwork
from torch_rl.noise.ou import OUNoise
from torch_rl.replay.continuous import ContinuousReplayBuffer


class Agent:
    def __init__(self, input_dims, action_space, tau, fc_dims, actor_optimizer_type, critic_optimizer_type,
                 actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
                 max_size=1000000, batch_size=64, randomized=False):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.noise = OUNoise(action_space, mu=np.zeros(action_space.shape))

        self.memory = ContinuousReplayBuffer(max_size, input_dims, action_space.shape[0], randomized)

        self.actor = PolicyNetwork(input_dims, action_space.shape[0], fc_dims, actor_optimizer_type,
                                   actor_optimizer_args)

        self.critic = ValueNetwork(input_dims, action_space.shape, fc_dims, critic_optimizer_type,
                                   critic_optimizer_args)

        self.target_actor = PolicyNetwork(input_dims, action_space.shape[0], fc_dims, actor_optimizer_type,
                                          actor_optimizer_args)

        self.target_critic = ValueNetwork(input_dims, action_space.shape, fc_dims, critic_optimizer_type,
                                          critic_optimizer_args)

        self.update_network_parameters(soft_tau=1)

        self.policy_loss_history = []
        self.value_loss_history = []

    def update_network_parameters(self, soft_tau=None):
        if soft_tau is None:
            soft_tau = self.tau

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
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

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = \
            self.memory.sample_buffer(self.batch_size)

        masks = 1 - int(done)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        masks = T.tensor(masks).to(self.actor.device)

        self.actor.optimizer.zero_grad()

        policy_loss = self.critic.forward(states, self.actor.forward(states))
        policy_loss = -policy_loss.mean()

        policy_loss.backward()

        self.policy_loss_history.append(policy_loss.item())

        self.actor.optimizer.step()

        self.critic.optimizer.zero_grad()

        target_actions = self.target_actor.forward(states_)
        target_value = self.target_critic.forward(states_, target_actions)

        expected_value = rewards + (masks * self.gamma * target_value)
        value = self.critic.forward(states, actions)

        value_loss = F.mse_loss(expected_value, value)

        value_loss.backward()

        self.value_loss_history.append(value_loss.item())

        self.critic.optimizer.step()

        self.update_network_parameters()
