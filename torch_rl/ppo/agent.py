import numpy as np
import torch as T
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from torch_rl.utils.utils import have_we_ran_out_of_time

from .network import PolicyNetwork, ValueNetwork


class Agent:
    def __init__(self, env, fc_dims, actor_optimizer_type, critic_optimizer_type,
                 actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
                 updates_per_iteration=5, batch_size=64, clip=0.2):

        self.env = env
        self.actor = PolicyNetwork(env.observation_space.shape, env.action_space.shape[0], fc_dims,
                                   actor_optimizer_type, actor_optimizer_args)
        self.critic = ValueNetwork(env.observation_space.shape, fc_dims, critic_optimizer_type, critic_optimizer_args)

        self.gamma = gamma
        self.updates_per_iteration = updates_per_iteration
        self.batch_size = batch_size
        self.clip = clip

        self.cov_var = T.full(size=env.action_space.shape, fill_value=0.5)
        self.cov_mat = T.diag(self.cov_var)

        self.actor_loss_history = []
        self.critic_loss_history = []

        self.scores_train = []
        self.num_time_steps_train = 0

    def learn(self, total_time_steps):

        while self.num_time_steps_train < total_time_steps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            self.num_time_steps_train += np.sum(batch_lens)

            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = T.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = T.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-T.min(surr1, surr2)).mean()
                critic_loss = F.mse_loss(V, batch_rtgs)

                self.actor.optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor.optimizer.step()

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                self.actor_loss_history.append(abs(actor_loss.item()))
                self.critic_loss_history.append(critic_loss.item())

    def test(self, n_games):
        scores = np.zeros(n_games)
        num_time_steps = 0

        for i in range(n_games):
            score = 0
            obs = self.env.reset()
            done = False
            t = 0
            while not done and not have_we_ran_out_of_time(self.env, t):
                action, _ = self.get_action(obs, train=False)
                obs, rew, done, _ = self.env.step(action)
                score += rew
                t += 1
            scores[i] = score
            num_time_steps += t

        return num_time_steps, np.mean(scores)

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []

        t = 0

        while t < self.batch_size:
            ep_rews = []
            score = 0
            obs = self.env.reset()
            done = False
            ep_t = 0
            while not done and not have_we_ran_out_of_time(self.env, ep_t):
                t += 1

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs, train=True)
                obs, rew, done, _ = self.env.step(action)

                score += rew

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                ep_t += 1

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

            self.scores_train.append(score)

        batch_obs = T.tensor(batch_obs, dtype=T.float)
        batch_acts = T.tensor(batch_acts, dtype=T.float)
        batch_log_probs = T.tensor(batch_log_probs, dtype=T.float)
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        return T.tensor(batch_rtgs, dtype=T.float)

    def get_action(self, obs, train=True):
        obs = T.tensor(obs, dtype=T.float)
        if train:
            mean = self.actor.forward(obs)
            dist = MultivariateNormal(mean, self.cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.cpu().detach().numpy(), log_prob.cpu().detach()
        else:
            return self.actor.forward(obs), None

    def evaluate(self, batch_obs, batch_acts):
        batch_obs = T.tensor(batch_obs, dtype=T.float)
        V = self.critic.forward(batch_obs).squeeze()
        mean = self.actor.forward(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs
