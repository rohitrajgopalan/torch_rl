import numpy as np

from .agent import Agent


def run(env, n_games, fc_dims, actor_optimizer_type, critic_optimizer_type,
        actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
        updates_per_iteration=5, batch_size=64, clip=0.2):
    agent = Agent(env, fc_dims, actor_optimizer_type, critic_optimizer_type,
                  actor_optimizer_args, critic_optimizer_args, gamma, updates_per_iteration, batch_size, clip)

    if type(n_games) == int:
        n_games_train = n_games
        n_games_test = n_games
    elif type(n_games) == tuple:
        n_games_train, n_games_test = n_games
    else:
        raise ValueError('n_games should either be int or tuple')

    agent.learn(n_games_train * env._max_episode_steps)

    num_time_steps_test, avg_score_test = agent.test(n_games_test)

    return agent.num_time_steps_train, np.mean(agent.scores_train), num_time_steps_test, avg_score_test, \
           np.mean(agent.actor_loss_history), np.mean(agent.critic_loss_history)