import numpy as np

from .agent import Agent


def run(env, n_games, gamma, tau,
        mem_size, batch_size, fc_dims, optimizer_type, optimizer_args={}, randomized=False, clip_param=0.2):
    agent = Agent(gamma, tau, env.action_space.shape, env.observation_space.shape,
                  mem_size, batch_size, fc_dims, optimizer_type, optimizer_args, randomized, clip_param)

    scores = np.zeros(n_games)

    for i in range(n_games):
        score = 0
        observation = env.reset()
        done = False
        t = 0
        while not done:
            action, log_prob, value = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, log_prob, reward, value, 1-int(done))
            agent.learn()
            observation = observation_

            if hasattr(env, '_max_episode_steps') and t == env._max_episode_steps:
                break

            t += 1

        scores[i] = score

    return np.mean(scores), np.mean(agent.loss_history)
