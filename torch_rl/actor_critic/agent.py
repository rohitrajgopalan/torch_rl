import torch as T
import torch.nn.functional as F

from .network import Network


class Agent:
    def __init__(self, gamma, n_actions, input_dims,
                 fc_dims, optimizer_type, optimizer_args={}):
        self.gamma = gamma
        self.actor_critic = Network(input_dims, n_actions, fc_dims, optimizer_type, optimizer_args)
        self.log_prob = 0
        self.actor_loss_history = []
        self.critic_loss_history = []

    def choose_action(self, observation, train=True):
        if train:
            state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
            probabilities, _ = self.actor_critic.forward(state)
            probabilities = F.softmax(probabilities, dim=1)
            action_probs = T.distributions.Categorical(probabilities)
            action = action_probs.sample()
            log_prob = action_probs.log_prob(action)
            self.log_prob = log_prob

            return action.item()
        else:
            state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
            probabilities, _ = self.actor_critic.forward(state)
            probabilities = F.softmax(probabilities, dim=1)

            return T.argmax(probabilities).item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()

        self.actor_loss_history.append(abs(actor_loss.item()))
        self.critic_loss_history.append(critic_loss.item())
