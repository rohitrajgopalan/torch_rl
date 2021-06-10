import numpy as np

from .agent import Agent
from torch_rl.utils.utils import have_we_ran_out_of_time


def run(env, n_games, tau, fc_dims, actor_optimizer_type, critic_optimizer_type,
        actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
        max_size=1000000, batch_size=64, randomized=False, policy_update_interval=2, noise_std=0.2,
        noise_clip=0.5):
    agent = Agent(env.observation_space.shape, env.action_space, tau, fc_dims, actor_optimizer_type,
                  critic_optimizer_type, actor_optimizer_args, critic_optimizer_args, gamma, max_size, batch_size,
                  randomized, policy_update_interval, noise_std, noise_clip)

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

            agent.remember(observation, action, reward, observation_, done)
            agent.learn()

            observation = observation_

            t += 1

        scores[i] = score
        num_time_steps += t

    return num_time_steps, np.mean(scores), np.mean(agent.policy_loss_history), np.mean(agent.value1_loss_history), np.mean(agent.value2_loss_history)
