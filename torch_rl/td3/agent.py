import torch as T
import torch.nn.functional as F

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

        target_actions = self.target_actor.forward(states_)
        noise = T.normal(T.zeros(target_actions.size()), self.noise_std).to(self.actor.device)
        noise = T.clamp(noise, -self.noise_clip, self.noise_clip)
        target_actions += noise

        target_q_value1 = self.target_critic1.forward(states_, target_actions)
        target_q_value2 = self.target_critic2.forward(states_, target_actions)
        target_q_value = T.min(target_q_value1, target_q_value2)
        expected_q_value = rewards + (masks * self.gamma * target_q_value)

        q_value1 = self.critic1.forward(states, actions)
        q_value2 = self.critic2.forward(states, actions)

        value_loss1 = F.mse_loss(q_value1, expected_q_value)
        value_loss2 = F.mse_loss(q_value2, expected_q_value)

        self.critic1.optimizer.zero_grad()
        value_loss1.backward()
        self.critic1.optimizer.step()

        self.critic2.optimizer.zero_grad()
        value_loss2.backward()
        self.critic2.optimizer.step()

        if self.learn_step_cntr % self.policy_update_interval == 0:
            policy_loss = self.critic1.forward(states, self.actor.forward(states))
            policy_loss = -T.mean(policy_loss)

            self.actor.optimizer.zero_grad()
            policy_loss.backward()
            self.actor.optimizer.step()

            self.policy_loss_history.append(abs(policy_loss.item()))

            self.update_network_parameters()

        self.value1_loss_history.append(value_loss1.item())
        self.value2_loss_history.append(value_loss2.item())
