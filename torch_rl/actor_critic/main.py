import numpy as np

from .agent import Agent
from torch_rl.utils.utils import have_we_ran_out_of_time


def run(env, n_games, gamma, fc_dims, optimizer_type, optimizer_args={}):
    agent = Agent(gamma, env.action_space.n, env.observation_space.shape,
                  fc_dims, optimizer_type, optimizer_args)

    scores = np.zeros(n_games)

    num_time_steps = 0

    for i in range(n_games):
        score = 0
        observation = env.reset()
        done = False

        t = 0
        while not done and not have_we_ran_out_of_time(env, t):
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward

            agent.learn(observation, reward, observation_, done)

            observation = observation_

            t += 1

        scores[i] = score
        num_time_steps += t

    return num_time_steps, np.mean(scores), np.mean(agent.actor_loss_history), np.mean(agent.critic_loss_history)