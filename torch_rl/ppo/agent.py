import torch as T

from .network import Network
from .buffer import ReplayBuffer

class Agent:
    def __init__(self, gamma, tau, action_dim, input_dims,
                 mem_size, batch_size, fc_dims, optimizer_type, optimizer_args={}, randomized=False, clip_param=0.2):
        self.input_dims = input_dims
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.clip = clip_param
        self.batch_size = batch_size

        self.memory = ReplayBuffer(mem_size, input_dims, action_dim[0], randomized)
        self.network = Network(input_dims, action_dim[0], fc_dims, optimizer_type, optimizer_args)

        self.loss_history = []

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.network.device)

        dist, value = self.network.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy(), value.cpu().detach().numpy()

    def store_transition(self, state, action, log_prob, reward, value, mask):
        self.memory.store_transition(state, action, log_prob, reward, value, mask)

    def compute_gae(self, rewards, masks, values):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def learn(self):
        self.network.optimizer.zero_grad()

        states, actions, log_probs, rewards, values, masks = self.memory.sample_buffer(self.batch_size)

        returns = self.compute_gae(rewards, masks, values)

        states = T.tensor(states).to(self.network.device)
        actions = T.tensor(actions).to(self.network.device)
        old_log_probs = T.tensor(log_probs).to(self.network.device)
        returns = T.tensor(returns, dtype=T.float32).to(self.network.device)
        values = T.tensor(values).to(self.network.device)

        advantages = returns - values

        dist, value = self.network.forward(states)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(actions)

        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = T.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages

        actor_loss = - T.min(surr1, surr2).mean()
        critic_loss = (returns - value).pow(2).mean()

        loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
        loss.backward()

        self.network.optimizer.step()
        self.loss_history.append(loss.item())
