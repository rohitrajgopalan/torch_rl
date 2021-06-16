import torch as T
import torch.nn.functional as F

from torch_rl.replay.continuous import ContinuousReplayBuffer
from .network import ActorNetwork, CriticNetwork, ValueNetwork


class Agent:
    def __init__(self, input_dims, action_space, tau,
                 fc_dims, actor_optimizer_type, critic_optimizer_type, value_optimizer_type,
                 actor_optimizer_args={}, critic_optimizer_args={},
                 value_optimizer_args={}, gamma=0.99, max_size=1000000, batch_size=100, reward_scale=2,
                 randomized=False):

        self.gamma = gamma
        self.tau = tau
        self.memory = ContinuousReplayBuffer(max_size, input_dims, action_space.shape[0], randomized)
        self.batch_size = batch_size

        self.actor = ActorNetwork(input_dims, fc_dims, action_space.high, action_space.shape[0],
                                  actor_optimizer_type, actor_optimizer_args)

        self.critic1 = CriticNetwork(input_dims, fc_dims, action_space.shape[0], critic_optimizer_type,
                                     critic_optimizer_args)
        self.critic2 = CriticNetwork(input_dims, fc_dims, action_space.shape[0], critic_optimizer_type,
                                     critic_optimizer_args)

        self.value = ValueNetwork(input_dims, fc_dims, value_optimizer_type, value_optimizer_args)
        self.target_value = ValueNetwork(input_dims, fc_dims, value_optimizer_type, value_optimizer_args)

        self.scale = reward_scale
        self.update_network_parameters(soft_tau=1)

        self.actor_loss_history = []
        self.critic_loss_history = []

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic1.device)
        done = T.tensor(done).to(self.critic1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic1.device)

        value = self.value.forward(state).view(-1)
        value_ = self.target_value.forward(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state, actions)
        q2_new_policy = self.critic2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * (F.mse_loss(value, value_target))
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state, actions)
        q2_new_policy = self.critic2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic1.forward(state, action).view(-1)
        q2_old_policy = self.critic2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()
        self.update_network_parameters()

        self.actor_loss_history.append(abs(actor_loss.item()))
        self.critic_loss_history.append(critic_loss.item())

    def update_network_parameters(self, soft_tau=None):
        if soft_tau is None:
            soft_tau = self.tau

        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
