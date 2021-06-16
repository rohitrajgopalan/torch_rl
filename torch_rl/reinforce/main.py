import numpy as np

from .agent import Agent
from torch_rl.utils.utils import have_we_ran_out_of_time


def run(env, n_games, gamma, fc_dims, optimizer_type, optimizer_args={}):
    agent = Agent(gamma, env.action_space.n, env.observation_space.shape,
                  fc_dims, optimizer_type, optimizer_args)

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
        score = 0
        observation = env.reset()
        done = False

        t = 0
        while not done and not have_we_ran_out_of_time(env, t):
            action = agent.choose_action(observation, train=True)
            observation_, reward, done, _ = env.step(action)
            score += reward

            agent.store_rewards(reward)
            agent.learn()

            observation = observation_

            t += 1

        scores_train[i] = score
        num_time_steps_train += t

    if n_games_test == 0:
        return num_time_steps_train, np.mean(scores_train), 0, -1, np.mean(agent.loss_history)
    else:
        scores_test = np.zeros(n_games_test)
        num_time_steps_test = 0

        for i in range(n_games_test):
            score = 0
            observation = env.reset()
            done = False

            t = 0
            while not done and not have_we_ran_out_of_time(env, t):
                action = agent.choose_action(observation, train=False)
                observation, reward, done, _ = env.step(action)
                score += reward
                t += 1

            scores_test[i] = score
            num_time_steps_test += t

        return num_time_steps_train, np.mean(scores_train), num_time_steps_test, np.mean(scores_test), np.mean(agent.loss_history)
