import numpy as np
from gym.spaces import Box

from torch_rl.utils.utils import have_we_ran_out_of_time, develop_memory_for_dt_action_blocker
from torch_rl.action_blocker.dt_action_blocker import DTActionBlocker
from .agent import DuelingTDAgent


def create_agent(env, algorithm_type, is_double, gamma, mem_size, batch_size, network_args,
                 optimizer_type, policy_type, policy_args={},
                 replace=1000, optimizer_args={}, enable_action_blocking=False,
                 min_penalty=0, goal=None, assign_priority=False, use_ml_for_action_blocking=False,
                 action_blocking_model_name=None):
    input_dims = env.observation_space.shape if type(env.observation_space) == Box else (1,)

    if use_ml_for_action_blocking:
        agent = DuelingTDAgent(algorithm_type, is_double, gamma, env.action_space.n, input_dims,
                               mem_size, batch_size, network_args, optimizer_type, policy_type, policy_args,
                               replace, optimizer_args, False, 0, goal, assign_priority)
        memory = develop_memory_for_dt_action_blocker(env)
        action_blocker = DTActionBlocker(env.observation_space, model_name=action_blocking_model_name,
                                         memory=memory, penalty=min_penalty)
        agent.action_blocker = action_blocker
        agent.enable_action_blocking = True
    else:
        agent = DuelingTDAgent(algorithm_type, is_double, gamma, env.action_space, input_dims,
                               mem_size, batch_size, network_args, optimizer_type, policy_type, policy_args,
                               replace, optimizer_args, enable_action_blocking, min_penalty, goal, assign_priority)

    return agent


def run(env, n_games, algorithm_type, is_double, gamma, mem_size, batch_size, network_args,
        optimizer_type, policy_type, policy_args={},
        replace=1000, optimizer_args={}, enable_action_blocking=False,
        min_penalty=0, goal=None, assign_priority=False, use_ml_for_action_blocking=False,
        action_blocking_model_name=None):

    agent = create_agent(env, algorithm_type, is_double, gamma, mem_size, batch_size, network_args, optimizer_type,
                         policy_type, policy_args, replace, optimizer_args, enable_action_blocking, min_penalty,
                         goal, assign_priority, use_ml_for_action_blocking, action_blocking_model_name)

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
            action = agent.choose_action(env, observation, True)
            observation_, reward, done, _ = env.step(action)

            if enable_action_blocking and agent.initial_action_blocked and reward > 0:
                reward *= -1
            score += reward

            agent.store_transition(observation, agent.initial_action,
                                   reward, observation_, done)
            agent.learn()

            observation = observation_

            t += 1

        scores_train[i] = score
        num_time_steps_train += t

    if n_games_test == 0:
        if enable_action_blocking:
            num_actions_blocked = agent.action_blocker.num_actions_blocked
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
            num_actions_blocked_train = agent.action_blocker.num_actions_blocked
            agent.action_blocker.num_actions_blocked = 0
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
                action = agent.choose_action(env, observation, False)
                observation_, reward, done, _ = env.step(action)
                score += reward

                t += 1

            scores_test[i] = score
            num_time_steps_test += t

        if enable_action_blocking:
            num_actions_blocked_test = agent.action_blocker.num_actions_blocked
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
