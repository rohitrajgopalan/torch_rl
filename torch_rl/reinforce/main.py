import numpy as np

from .agent import Agent


def run(env, n_games, gamma, fc1_dims, fc2_dims, optimizer_type, optimizer_args={}):
    agent = Agent(gamma, env.action_space.n, env.observation_space.shape,
                  fc1_dims, fc2_dims, optimizer_type, optimizer_args)

    scores = np.zeros(n_games)

    for i in range(n_games):
        score = 0
        observation = env.reset()
        done = False

        t = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward

            agent.store_rewards(reward)
            agent.learn()

            observation = observation_

            if hasattr(env, '_max_episode_steps') and t == env._max_episode_steps:
                break

            t += 1

        scores[i] = score

    return np.mean(scores), np.mean(agent.loss_history)
