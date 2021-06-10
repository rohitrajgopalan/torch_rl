import numpy as np

from .agent import Agent
from torch_rl.utils.utils import have_we_ran_out_of_time


def run(env, n_games, gamma, tau,
        mem_size, batch_size, fc_dims, optimizer_type, optimizer_args={}, randomized=False, clip_param=0.2):
    agent = Agent(gamma, tau, env.action_space.shape, env.observation_space.shape,
                  mem_size, batch_size, fc_dims, optimizer_type, optimizer_args, randomized, clip_param)

    scores = np.zeros(n_games)

    num_time_steps = 0

    for i in range(n_games):
        score = 0
        observation = env.reset()
        done = False
        t = 0
        while not done and not have_we_ran_out_of_time(env, t):
            action, log_prob, value = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, log_prob, reward, value, 1 - int(done))
            agent.learn()
            observation = observation_

            t += 1

        scores[i] = score
        num_time_steps += t

    return num_time_steps, np.mean(scores), np.mean(agent.loss_history)
