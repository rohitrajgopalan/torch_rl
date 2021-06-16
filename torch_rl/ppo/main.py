import numpy as np

from .agent import Agent


def run(env, n_games, fc_dims, actor_optimizer_type, critic_optimizer_type,
        actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
        updates_per_iteration=5, batch_size=64, clip=0.2):
    agent = Agent(env, fc_dims, actor_optimizer_type, critic_optimizer_type,
                  actor_optimizer_args, critic_optimizer_args, gamma, updates_per_iteration, batch_size, clip)

    max_time_steps = n_games * env._max_episode_steps

    agent.learn(max_time_steps)

    return np.mean(agent.scores), np.mean(agent.actor_loss_history), np.mean(agent.critic_loss_history)