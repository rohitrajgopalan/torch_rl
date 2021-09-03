import numpy as np
import torch as T
import torch.nn.functional as F
from .network import PolicyNetwork, ValueNetwork
from torch_rl.noise.ou import OUNoise
from torch_rl.replay.replay import ReplayBuffer
from torch_rl.replay.priority_replay import PriorityReplayBuffer


class DDPGAgent:
    def __init__(self, input_dims, action_space, tau, network_args, actor_optimizer_type, critic_optimizer_type,
                 actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
                 max_size=1000000, batch_size=64, goal=None, assign_priority=False, model_name=None):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.noise = OUNoise(action_space, mu=np.zeros(action_space.shape))

        self.goal = goal
        if self.goal is not None:
            if not type(self.goal) == np.ndarray:
                self.goal = np.array([self.goal]).astype(np.float32)
            self.actor = PolicyNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                       action_space.shape[0], network_args, actor_optimizer_type,
                                       actor_optimizer_args)

            self.critic = ValueNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                       action_space.shape, network_args, critic_optimizer_type,
                                       critic_optimizer_args)

            self.target_actor = PolicyNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                              action_space.shape[0], network_args, actor_optimizer_type,
                                              actor_optimizer_args)

            self.target_critic = ValueNetwork(tuple(np.add(input_dims, self.goal.shape)),
                                              action_space.shape, network_args, critic_optimizer_type,
                                              critic_optimizer_args)
        else:
            self.actor = PolicyNetwork(input_dims, action_space.shape[0], network_args, actor_optimizer_type,
                                       actor_optimizer_args)

            self.critic = ValueNetwork(input_dims, action_space.shape, network_args, critic_optimizer_type,
                                       critic_optimizer_args)

            self.target_actor = PolicyNetwork(input_dims, action_space.shape[0], network_args, actor_optimizer_type,
                                              actor_optimizer_args)

            self.target_critic = ValueNetwork(input_dims, action_space.shape, network_args, critic_optimizer_type,
                                              critic_optimizer_args)

        self.update_network_parameters(soft_tau=1)

        if assign_priority:
            self.memory = PriorityReplayBuffer(max_size, input_dims, action_space.shape[0], self.goal)
        else:
            self.memory = ReplayBuffer(max_size, input_dims, action_space.shape[0], self.goal)

        self.action_space = action_space

        if model_name is not None:
            self.load_model(model_name)

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
            goal = T.tensor([self.goal], dtype=T.float).to(
                self.target_actor.device if use_target else self.actor.device)
            inputs = T.cat([state, goal], dim=1)
        else:
            inputs = state
        action = T.tensor([action], dtype=T.float).to(self.target_actor.device if use_target else self.actor.device)
        if use_target:
            values = self.target_critic.forward(inputs, action).cpu().detach().numpy()
        else:
            values = self.critic.forward(inputs, action).cpu().detach().numpy()
        return values.squeeze().item()

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
        states = T.tensor(state, dtype=T.float).to(self.critic.device)
        actions = T.tensor(action, dtype=T.float).to(self.critic.device)
        rewards = T.tensor(reward, dtype=T.float).to(self.critic.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.critic.device)
        dones = T.tensor(done).to(self.critic.device)

        if goals is not None:
            goals = T.tensor(goals).to(self.critic.device)

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

    def load_model(self, model_name):
        self.actor.load_model('{0}_actor'.format(model_name))
        self.target_actor.load_model('{0}_target_actor'.format(model_name))
        self.critic.load_model('{0}_critic'.format(model_name))
        self.target_critic.load_model('{0}_target_critic'.format(model_name))

    def save_model(self, model_name):
        self.actor.save_model('{0}_actor'.format(model_name))
        self.target_actor.save_model('{0}_target_actor'.format(model_name))
        self.critic.save_model('{0}_critic'.format(model_name))
        self.target_critic.save_model('{0}_target_critic'.format(model_name))

    def apply_transfer_learning(self, td3_actor_model, td3_critic_model,
                                td3_target_actor_model=None, td3_target_critic_model=None):
        self.actor.load_model(td3_actor_model)
        self.critic.load_model(td3_critic_model)
        if td3_target_actor_model is not None:
            self.target_actor.load_model(td3_target_actor_model)
        if td3_target_critic_model is not None:
            self.target_critic.load_model(td3_target_critic_model)

    def __str__(self):
        return "DDPG Agent"
