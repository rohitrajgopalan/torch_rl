import numpy as np

from .agent import Agent
from torch_rl.utils.utils import have_we_ran_out_of_time


def run(env, n_games, tau, fc_dims, actor_optimizer_type, critic_optimizer_type,
        actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
        max_size=1000000, batch_size=64, randomized=False, policy_update_interval=2, noise_std=0.2,
        noise_clip=0.5):
    agent = Agent(env.observation_space.shape, env.action_space, tau, fc_dims, actor_optimizer_type,
                  critic_optimizer_type, actor_optimizer_args, critic_optimizer_args, gamma, max_size, batch_size,
                  policy_update_interval, noise_std, noise_clip)

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

            agent.remember(observation, action, reward, observation_, done)
            agent.learn()

            observation = observation_

            t += 1

        scores_train[i] = score
        num_time_steps_train += t

    if n_games_test == 0:
        return num_time_steps_train, np.mean(scores_train), np.mean(agent.policy_loss_history), np.mean(agent.value1_loss_history), np.mean(agent.value2_loss_history)
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

        return num_time_steps_train, np.mean(scores_train), num_time_steps_test, np.mean(scores_test), \
               np.mean(agent.policy_loss_history), np.mean(
            agent.value1_loss_history), np.mean(agent.value2_loss_history)
