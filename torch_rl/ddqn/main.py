import numpy as np

from .agent import Agent


def run(env, n_games, gamma, epsilon,
        mem_size, batch_size, fc1_dims, fc2_dims, optimizer_type, eps_min=0.01, eps_dec=5e-7,
        replace=1000, optimizer_args={}, randomized=False):
    agent = Agent(gamma, epsilon, env.action_space.n, env.observation_space.shape,
                  mem_size, batch_size, fc1_dims, fc2_dims, optimizer_type, eps_min, eps_dec,
                  replace, optimizer_args, randomized)

    scores = np.zeros(n_games)

    for i in range(n_games):
        score = 0
        observation = env.reset()
        done = False

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward

            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()

            observation = observation_

        scores[i] = score

    return np.mean(scores), np.mean(agent.loss_history)
