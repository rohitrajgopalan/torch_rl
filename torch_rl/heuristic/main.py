from torch_rl.utils.utils import get_next_discrete_action_heuristic, have_we_ran_out_of_time
from torch_rl.utils.types import LearningType
from .heuristic_with_dt import HeuristicWithDT
from .heuristic_with_td import HeuristicWithTD
from .heuristic_with_dueling_td import HeuristicWithDuelingTD
from torch_rl.action_blocker.action_blocker import ActionBlocker

import numpy as np
from gym.spaces import Discrete, Box


def run_with_dt(env, n_games, learning_type, heuristic_func,
                use_model_only, enable_action_blocking=False, min_penalty=0):
    assert learning_type == LearningType.OFFLINE
    agent = HeuristicWithDT(heuristic_func, use_model_only, type(env.action_space) == Box)
    return run(env, n_games, agent, learning_type, enable_action_blocking, min_penalty)


def run_with_td(env, n_games, learning_type, heuristic_func, use_model_only,
                algorithm_type, is_double, gamma, mem_size, batch_size, fc_dims,
                optimizer_type, policy_type, policy_args={},
                replace=1000, optimizer_args={}, enable_action_blocking=False,
                min_penalty=0, goal=None):
    agent = HeuristicWithTD(heuristic_func, use_model_only, algorithm_type, is_double, gamma, mem_size, batch_size,
                            fc_dims, optimizer_type, policy_type, policy_args, replace, optimizer_args,
                            enable_action_blocking, min_penalty, goal)
    return run(env, n_games, agent, learning_type, enable_action_blocking, min_penalty)


def run_with_dueling_td(env, n_games, learning_type, heuristic_func, use_model_only,
                        algorithm_type, is_double, gamma, mem_size, batch_size, fc_dims,
                        optimizer_type, policy_type, policy_args={},
                        replace=1000, optimizer_args={}, enable_action_blocking=False,
                        min_penalty=0, goal=None):
    agent = HeuristicWithDuelingTD(heuristic_func, use_model_only, algorithm_type, is_double, gamma, mem_size,
                                   batch_size,
                                   fc_dims, optimizer_type, policy_type, policy_args, replace, optimizer_args,
                                   enable_action_blocking, min_penalty, goal)
    return run(env, n_games, agent, learning_type, enable_action_blocking, min_penalty)


def run(env, n_games, agent, learning_type, enable_action_blocking=False, min_penalty=0):
    learning_types = [LearningType.OFFLINE, LearningType.ONLINE] if learning_type == LearningType.BOTH else [
        learning_type]
    if type(n_games) == int:
        n_games_train = n_games
        n_games_test = n_games
    elif type(n_games) == tuple:
        n_games_train, n_games_test = n_games
    else:
        raise ValueError('n_games should either be int or tuple')

    action_blocker = None
    if enable_action_blocking and type(env.action_space) == Discrete:
        action_blocker = ActionBlocker(env.action_space, min_penalty)

    scores_train = np.zeros((n_games_train * len(learning_types)))
    num_time_steps_train = 0
    for current_learning_type in learning_types:
        for i in range(n_games_train):
            score = 0
            observation = env.reset()
            done = False

            t = 0
            while not done and not have_we_ran_out_of_time(env, t):
                if type(env.action_space) == Discrete:
                    original_action, action_blocked, action = get_next_discrete_action_heuristic(agent,
                                                                                                 current_learning_type,
                                                                                                 env, observation,
                                                                                                 True,
                                                                                                 enable_action_blocking,
                                                                                                 action_blocker)
                else:
                    original_action = agent.get_action(env, current_learning_type, observation, True)
                    action_blocked = False
                    action = original_action

                if action is None:
                    print('WARNING: No valid policy action found, running original action')
                    action = original_action

                observation_, reward, done, _ = env.step(action)

                if enable_action_blocking and action_blocked and reward > 0:
                    reward *= -1
                score += reward

                agent.store_transition(observation, original_action, reward, observation_, done)

                observation = observation_

                t += 1

            scores_train[i] = score
            num_time_steps_train += t
        agent.optimize(env, current_learning_type)
    if n_games_test == 0:
        if enable_action_blocking:
            num_actions_blocked = action_blocker.num_actions_blocked
        else:
            num_actions_blocked = 0

        return {
            'num_time_steps_train': num_time_steps_train,
            'avg_score_train': np.mean(scores_train),
            'num_actions_blocked_train': num_actions_blocked,
            'num_heuristic_actions_chosen_train': agent.num_heuristic_actions_chosen,
            'num_predicted_actions_chosen_train': agent.num_predicted_actions_chosen,
            'num_time_steps_test': 0,
            'avg_score_test': -1,
            'num_actions_blocked_test': 0,
            'num_heuristic_actions_chosen_test': 0,
            'num_predicted_actions_chosen_test': 0
        }
    else:
        if enable_action_blocking:
            num_actions_blocked_train = action_blocker.num_actions_blocked
            action_blocker.num_actions_blocked = 0
        else:
            num_actions_blocked_train = 0
        num_heuristic_actions_train = agent.num_heuristic_actions_chosen
        num_predicted_actions_train = agent.num_predicted_actions_chosen

        agent.reset_metrics()
        scores_test = np.zeros(n_games_test)
        num_time_steps_test = 0

        for i in range(n_games_test):
            score = 0
            observation = env.reset()
            done = False

            t = 0
            while not done and not have_we_ran_out_of_time(env, t):
                if type(env.action_space) == Discrete:
                    original_action, _, action = get_next_discrete_action_heuristic(agent,
                                                                                    learning_type,
                                                                                    env, observation,
                                                                                    False,
                                                                                    enable_action_blocking,
                                                                                    action_blocker)
                else:
                    original_action = agent.get_action(env, learning_type, observation, True)
                    action = original_action

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
            'num_heuristic_actions_chosen_train': num_heuristic_actions_train,
            'num_predicted_actions_chosen_train': num_predicted_actions_train,
            'num_time_steps_test': num_time_steps_test,
            'avg_score_test': np.mean(scores_test),
            'num_actions_blocked_test': num_actions_blocked_test,
            'num_heuristic_actions_chosen_test': agent.num_heuristic_actions_chosen,
            'num_predicted_actions_chosen_test': agent.num_predicted_actions_chosen
        }
