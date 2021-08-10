import numpy as np

from .agent import Agent
from torch_rl.utils.utils import have_we_ran_out_of_time


def run(env, n_games, tau, fc_dims, actor_optimizer_type, critic_optimizer_type,
        actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
        max_size=1000000, batch_size=64, goal=None):
    agent = Agent(env.observation_space.shape, env.action_space, tau, fc_dims, actor_optimizer_type,
                  critic_optimizer_type, actor_optimizer_args, critic_optimizer_args, gamma, max_size,
                  batch_size, goal)

    if type(n_games) == int:
        n_games_train = n_games
        n_games_test = n_games
    elif type(n_games) == tuple:
        n_games_train, n_games_test = n_games
    else:
        raise ValueError('n_games should either be int or tuple')

    scores_train = np.zeros(n_games_train)

    num_time_steps_train = 0

    for i in range(n_games_train):
        score_train = 0
        observation = env.reset()
        done = False

        agent.noise.reset()

        t = 0
        while not done and not have_we_ran_out_of_time(env, t):
            action = agent.choose_action(observation, t, True)
            observation_, reward, done, _ = env.step(action)
            score_train += reward

            agent.remember(observation, action, reward, observation_, done)
            agent.learn()

            observation = observation_

            t += 1

        scores_train[i] = score_train
        num_time_steps_train += t

    if n_games_test == 0:
        return {
            'num_time_steps_train': num_time_steps_train,
            'avg_score_train': np.mean(scores_train),
            'num_time_steps_test': 0,
            'avg_score_test': -1,
            'avg_policy_loss': np.mean(agent.policy_loss_history),
            'avg_value_loss': np.mean(agent.value_loss_history),
        }
    else:
        scores_test = np.zeros(n_games_test)

        num_time_steps_test = 0

        for i in range(n_games_test):
            score_test = 0
            observation = env.reset()
            done = False

            t = 0
            while not done and not have_we_ran_out_of_time(env, t):
                action = agent.choose_action(observation, t, False)
                observation, reward, done, _ = env.step(action)
                score_test += reward
                t += 1

            scores_train[i] = score_test
            num_time_steps_train += t

        return {
            'num_time_steps_train': num_time_steps_train,
            'avg_score_train': np.mean(scores_train),
            'num_time_steps_test': num_time_steps_test,
            'avg_score_test': np.mean(scores_test),
            'avg_policy_loss': np.mean(agent.policy_loss_history),
            'avg_value1_loss': np.mean(agent.value_loss_history)
        }
