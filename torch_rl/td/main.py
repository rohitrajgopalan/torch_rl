import numpy as np
from gym.spaces import Box

from .agent import TDAgent
from torch_rl.action_blocker.action_blocker import ActionBlocker
from torch_rl.utils.utils import have_we_ran_out_of_time, get_next_discrete_action


def run(env, n_games, algorithm_type, is_double, gamma, mem_size, batch_size, fc_dims,
        optimizer_type, policy_type, policy_args={},
        replace=1000, optimizer_args={}, enable_action_blocking=False,
        min_penalty=0, goal=None):
    input_dims = env.observation_space.shape if type(env.observation_space) == Box else (1,)
    agent = TDAgent(algorithm_type, is_double, gamma, env.action_space.n, input_dims,
                    mem_size, batch_size, fc_dims, optimizer_type, policy_type, policy_args,
                    replace, optimizer_args, goal)

    if type(n_games) == int:
        n_games_train = n_games
        n_games_test = n_games
    elif type(n_games) == tuple:
        n_games_train, n_games_test = n_games
    else:
        raise ValueError('n_games should either be int or tuple')

    action_blocker = None
    if enable_action_blocking:
        action_blocker = ActionBlocker(env.action_space, min_penalty)

    scores_train = np.zeros(n_games_train)
    num_time_steps_train = 0
    for i in range(n_games_train):
        score = 0
        observation = env.reset()
        done = False

        t = 0
        while not done and not have_we_ran_out_of_time(env, t):
            original_action, action_blocked, action = get_next_discrete_action(agent, env, observation,
                                                                               True, enable_action_blocking,
                                                                               action_blocker)

            if action is None:
                print('WARNING: No valid policy action found, running original action')
                action = original_action

            observation_, reward, done, _ = env.step(action)

            if enable_action_blocking and action_blocked and reward > 0:
                reward *= -1
            score += reward

            agent.store_transition(observation, original_action, reward, observation_, done)
            agent.learn()

            observation = observation_

            t += 1

        scores_train[i] = score
        num_time_steps_train += t

    if n_games_test == 0:
        if enable_action_blocking:
            num_actions_blocked = action_blocker.num_actions_blocked
        else:
            num_actions_blocked = 0

        return {
            'num_time_steps_train': num_time_steps_train,
            'avg_score_train': np.mean(scores_train),
            'num_actions_blocked_train': num_actions_blocked,
            'num_time_steps_test': 0,
            'avg_score_test': -1,
            'num_actions_blocked_test': 0,
            'avg_loss': np.mean(agent.loss_history)
        }
    else:
        if enable_action_blocking:
            num_actions_blocked_train = action_blocker.num_actions_blocked
            action_blocker.num_actions_blocked = 0
        else:
            num_actions_blocked_train = 0

        scores_test = np.zeros(n_games_test)
        num_time_steps_test = 0

        for i in range(n_games_test):
            score = 0
            observation = env.reset()
            done = False

            t = 0
            while not done and not have_we_ran_out_of_time(env, t):
                original_action, action_blocked, action = get_next_discrete_action(agent, env, observation,
                                                                                   False, enable_action_blocking,
                                                                                   action_blocker)

                if action is None:
                    print('WARNING: No valid policy action found, running original action')
                    action = original_action

                observation_, reward, done, _ = env.step(action)
                score += reward

                t += 1

            scores_test[i] = score
            num_time_steps_test += t

        if enable_action_blocking:
            num_actions_blocked_test = action_blocker.num_actions_blocked
        else:
            num_actions_blocked_test = 0

        return {
            'num_time_steps_train': num_time_steps_train,
            'avg_score_train': np.mean(scores_train),
            'num_actions_blocked_train': num_actions_blocked_train,
            'num_time_steps_test': num_time_steps_test,
            'avg_score_test': np.mean(scores_test),
            'num_actions_blocked_test': num_actions_blocked_test,
            'avg_loss': np.mean(agent.loss_history)
        }
