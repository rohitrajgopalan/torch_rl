import numpy as np
from gym.spaces import Box

from .agent import Agent
from torch_rl.action_blocker.action_blocker import ActionBlocker
from torch_rl.utils.utils import have_we_ran_out_of_time, get_next_discrete_action


def run(env, n_games, gamma, epsilon,
        mem_size, batch_size, fc_dims, optimizer_type, eps_min=0.01, eps_dec=5e-7,
        replace=1000, optimizer_args={}, randomized=False, enable_action_blocking=False,
        min_penalty=0):
    input_dims = env.observation_space.shape if type(env.observation_space) == Box else (1,)
    agent = Agent(gamma, epsilon, env.action_space.n, input_dims,
                  mem_size, batch_size, fc_dims, optimizer_type, eps_min, eps_dec,
                  replace, optimizer_args, randomized)

    if type(n_games) == int:
        n_games_train = n_games
        n_games_test = n_games
    elif type(n_games) == tuple:
        n_games_train, n_games_test = n_games
    else:
        raise ValueError('n_games should either be int or tuple')

    action_blocker = None
    if enable_action_blocking:
        action_blocker = ActionBlocker(input_dims, env.action_space, fc_dims, optimizer_type,
                                       optimizer_args, randomized, mem_size, min_penalty)

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
                print('WARNING: No valid policy action found, running sample action')
                action = env.action_space.sample()

            observation_, reward, done, _ = env.step(action)

            if enable_action_blocking and action_blocked:
                reward *= -1
            score += reward

            agent.store_transition(observation, original_action, reward, observation_, done)
            agent.learn()
            if enable_action_blocking:
                action_blocker.learn()

            observation = observation_

            t += 1

        scores_train[i] = score
        num_time_steps_train += t

    if n_games_test == 0:
        if enable_action_blocking:
            precision, recall = action_blocker.get_precision_and_recall()
            num_actions_blocked = action_blocker.num_actions_blocked
        else:
            precision, recall = -1, -1
            num_actions_blocked = 0

        return {
            'num_time_steps_train': num_time_steps_train,
            'avg_score_train': np.mean(scores_train),
            'action_blocker_precision_train': precision,
            'action_blocker_recall_train': recall,
            'num_actions_blocked_train': num_actions_blocked,
            'num_time_steps_test': 0,
            'avg_score_test': -1,
            'action_blocker_precision_test': -1,
            'action_blocker_recall_test': -1,
            'num_actions_blocked_test': 0,
            'avg_loss': np.mean(agent.loss_history)
        }
    else:
        if enable_action_blocking:
            precision, recall = action_blocker.get_precision_and_recall()
            num_actions_blocked_train = action_blocker.num_actions_blocked
            action_blocker.reset_confusion_matrix()
        else:
            precision, recall = -1, -1
            num_actions_blocked_train = 0

        scores_test = np.zeros(n_games_test)
        num_time_steps_test = 0

        for i in range(n_games_test):
            score = 0
            observation = env.reset()
            done = False

            t = 0
            while not done and not have_we_ran_out_of_time(env, t):
                _, action_blocked, action = get_next_discrete_action(agent, env, observation,
                                                                     False, enable_action_blocking,
                                                                     action_blocker)

                if action is None:
                    print('WARNING: No valid policy action found, running sample action')
                    action = env.action_space.sample()

                observation_, reward, done, _ = env.step(action)

                if enable_action_blocking and action_blocked:
                    reward *= -1
                score += reward

                t += 1

            scores_test[i] = score
            num_time_steps_test += t

        if enable_action_blocking:
            test_precision, test_recall = action_blocker.get_precision_and_recall()
            num_actions_blocked_test = action_blocker.num_actions_blocked
        else:
            test_precision, test_recall = -1, -1
            num_actions_blocked_test = 0

        return {
            'num_time_steps_train': num_time_steps_train,
            'avg_score_train': np.mean(scores_train),
            'action_blocker_precision_train': precision,
            'action_blocker_recall_train': recall,
            'num_actions_blocked_train': num_actions_blocked_train,
            'num_time_steps_test': num_time_steps_test,
            'avg_score_test': np.mean(scores_test),
            'action_blocker_precision_test': test_precision,
            'action_blocker_recall_test': test_recall,
            'num_actions_blocked_test': num_actions_blocked_test,
            'avg_loss': np.mean(agent.loss_history)
        }
