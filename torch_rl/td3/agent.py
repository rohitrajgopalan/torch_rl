import torch as T
import torch.nn.functional as F
import numpy as np

from torch_rl.ddpg.network import PolicyNetwork, ValueNetwork
from torch_rl.noise.gaussian import GaussianExploration
from torch_rl.replay.priority_replay import PriorityReplayBuffer
from torch_rl.replay.replay import ReplayBuffer


class Agent:
    def __init__(self, input_dims, action_space, tau, network_args, actor_optimizer_type, critic_optimizer_type,
                 actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
                 max_size=1000000, batch_size=64, policy_update_interval=2, noise_std=0.2,
                 noise_clip=0.5, goal=None, assign_priority=False):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_update_interval = policy_update_interval
        self.noise_std = noise_std
        self.noise_clip = noise_clip

        self.noise = GaussianExploration(action_space)

        self.goal = goal
        if self.goal is not None:
            if not type(self.goal) == np.ndarray:
                self.goal = np.array([self.goal]).astype(np.float32)
            self.actor = PolicyNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                       action_space.shape[0], network_args, actor_optimizer_type,
                                       actor_optimizer_args)

            self.critic1 = ValueNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                        action_space.shape, network_args, critic_optimizer_type,
                                        critic_optimizer_args)

            self.critic2 = ValueNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                        action_space.shape, network_args, critic_optimizer_type,
                                        critic_optimizer_args)

            self.target_actor = PolicyNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                              action_space.shape[0], network_args, actor_optimizer_type,
                                              actor_optimizer_args)

            self.target_critic1 = ValueNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                               action_space.shape, network_args, critic_optimizer_type,
                                               critic_optimizer_args)
            self.target_critic2 = ValueNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                               action_space.shape, network_args, critic_optimizer_type,
                                               critic_optimizer_args)
        else:
            self.actor = PolicyNetwork(input_dims, action_space.shape[0], network_args, actor_optimizer_type,
                                       actor_optimizer_args)

            self.critic1 = ValueNetwork(input_dims, action_space.shape, network_args, critic_optimizer_type,
                                        critic_optimizer_args)

            self.critic2 = ValueNetwork(input_dims, action_space.shape, network_args, critic_optimizer_type,
                                        critic_optimizer_args)

            self.target_actor = PolicyNetwork(input_dims, action_space.shape[0], network_args, actor_optimizer_type,
                                              actor_optimizer_args)

            self.target_critic1 = ValueNetwork(input_dims, action_space.shape, network_args, critic_optimizer_type,
                                               critic_optimizer_args)
            self.target_critic2 = ValueNetwork(input_dims, action_space.shape, network_args, critic_optimizer_type,
                                               critic_optimizer_args)

        self.update_network_parameters(soft_tau=1)

        if assign_priority:
            self.memory = PriorityReplayBuffer(max_size, input_dims, action_space.shape[0], self.goal)
        else:
            self.memory = ReplayBuffer(max_size, input_dims, action_space.shape[0], self.goal)

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

    def choose_action(self, observation, t=0, train=True):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        if self.goal is not None:
            goal = T.tensor(self.goal, dtype=T.float).to(self.actor.device)
            inputs = T.cat([state, goal], dim=0)
        else:
            inputs = state
        action = self.actor.forward(inputs)
        action = action.cpu().detach().numpy()

        if train:
            return self.noise.get_action(action, t)
        else:
            return np.clip(action, self.action_space.low, self.action_space.high)

    def get_critic_value(self, observation, action, use_target=False):
        state = T.tensor([observation], dtype=T.float).to(self.target_actor.device if use_target else self.actor.device)
        if self.goal is not None:
            goal = T.tensor([self.goal], dtype=T.float).to(self.target_actor.device if use_target else self.actor.device)
            inputs = T.cat([state, goal], dim=1)
        else:
            inputs = state
        action = T.tensor([action], dtype=T.float).to(self.target_actor.device if use_target else self.actor.device)

        if use_target:
            values1 = self.target_critic1.forward(inputs, action).cpu().detach().numpy()
            values2 = self.target_critic2.forward(inputs, action).cpu().detach().numpy()
        else:
            values1 = self.critic1.forward(inputs, action).cpu().detach().numpy()
            values2 = self.critic2.forward(inputs, action).cpu().detach().numpy()

        value1 = values1.squeeze().item()
        value2 = values2.squeeze().item()

        return value1 if value1 < value2 else value2

    def store_transition(self, state, action, reward, state_, done, t=0):
        self.memory.store_transition(state, action, reward, state_, done,
                                     error_val=self.determine_error(state, action, reward, state_, done, t))

    def determine_error(self, state, action, reward, state_, done, t=0):
        done = int(done)
        target_action = self.choose_action(state_, t, True)
        next_critic_value = self.get_critic_value(state_, target_action, True)
        old_critic_value = self.get_critic_value(state, action, False)

        return reward + (self.gamma * next_critic_value * (1 - done)) - old_critic_value

    def sample_memory(self):
        state, action, reward, state_, done, goals = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state, dtype=T.float).to(self.actor.device)
        actions = T.tensor(action, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(reward, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.actor.device)
        dones = T.tensor(done).to(self.actor.device)

        if goals is not None:
            goals = T.tensor(goals).to(self.actor.device)

        return states, actions, rewards, states_, dones, goals

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones, goals = self.sample_memory()

        if goals is not None:
            goals = T.tensor(goals).to(self.actor.device)
            inputs = T.cat([states, goals], dim=1).float()
            inputs_ = T.cat([states_, goals], dim=1).float()
        else:
            inputs = states.float()
            inputs_ = states_.float()

        target_actions = self.target_actor.forward(inputs_)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=self.noise_clip)), -self.noise_clip,
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

        self.policy_loss_history.append(abs(actor_loss.item()))
        self.value1_loss_history.append(critic_loss1.item())
        self.value2_loss_history.append(critic_loss2.item())

    def load_model(self, model_name):
        self.actor.load_model('{0}_actor'.format(model_name))
        self.target_actor.load_model('{0}_target_actor'.format(model_name))
        self.critic1.load_model('{0}_critic'.format(model_name))
        self.critic2.load_model('{0}_critic'.format(model_name))
        self.target_critic1.load_model('{0}_target_critic'.format(model_name))
        self.target_critic2.load_model('{0}_target_critic'.format(model_name))

    def save_model(self, model_name):
        self.actor.save_model('{0}_actor'.format(model_name))
        self.target_actor.save_model('{0}_target_actor'.format(model_name))
        self.critic1.save_model('{0}_critic'.format(model_name))
        self.critic2.save_model('{0}_critic'.format(model_name))
        self.target_critic1.save_model('{0}_target_critic'.format(model_name))
        self.target_critic2.save_model('{0}_target_critic'.format(model_name))

    def apply_transfer_learning(self, ddpg_actor_model, ddpg_critic_model,
                                ddpg_target_actor_model=None, ddpg_target_critic_model=None):
        self.actor.load_model(ddpg_actor_model)
        self.critic1.load_model(ddpg_critic_model)
        self.critic2.load_model(ddpg_critic_model)
        if ddpg_target_actor_model is not None:
            self.target_actor.load_model(ddpg_target_actor_model)
        if ddpg_target_critic_model is not None:
            self.target_critic1.load_model(ddpg_target_critic_model)
            self.target_critic2.load_model(ddpg_target_critic_model)
