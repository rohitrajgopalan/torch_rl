import numpy as np
from gym.spaces import Discrete

from torch_rl.utils.types import LearningType
from torch_rl.utils.utils import have_we_ran_out_of_time, develop_memory_for_dt_action_blocker
from .heuristic_with_dt import HeuristicWithDT
from .heuristic_with_dueling_td import HeuristicWithDuelingTD
from .heuristic_with_td import HeuristicWithTD
from .heuristic_with_ddpg import HeuristicWithDDPG
from .heuristic_with_td3 import HeuristicWithTD3

from torch_rl.action_blocker.dt_action_blocker import DTActionBlocker


def run_with_dt(env, n_games, learning_type, heuristic_func,
                use_model_only, enable_action_blocking=False, min_penalty=0,
                use_ml_for_action_blocking=False, action_blocking_model_name=None, **args):
    assert learning_type == LearningType.OFFLINE
    if use_ml_for_action_blocking and type(env.action_space) == Discrete:
        agent = HeuristicWithDT(heuristic_func, use_model_only, env.action_space, False, 0, **args)
        memory = develop_memory_for_dt_action_blocker(env)
        action_blocker = DTActionBlocker(env.observation_space, model_name=action_blocking_model_name,
                                         memory=memory, penalty=min_penalty)
        agent.action_blocker = action_blocker
    else:
        agent = HeuristicWithDT(heuristic_func, use_model_only, env.action_space,
                                enable_action_blocking, min_penalty, **args)
    return run(env, n_games, agent, learning_type)


def run_with_td(env, n_games, learning_type, heuristic_func, use_model_only,
                algorithm_type, is_double, gamma, mem_size, batch_size, network_args,
                optimizer_type, policy_type, policy_args={},
                replace=1000, optimizer_args={}, enable_action_blocking=False,
                min_penalty=0, goal=None, add_conservative_loss=False, alpha=0.001,
                use_ml_for_action_blocking=False, action_blocking_model_name=None, **args):
    if use_ml_for_action_blocking:
        agent = HeuristicWithTD(heuristic_func, use_model_only, algorithm_type, is_double, gamma,
                                env.action_space, env.observation_space.shape, mem_size, batch_size,
                                network_args, optimizer_type, policy_type, policy_args, replace, optimizer_args,
                                False, 0, goal, add_conservative_loss, alpha, **args)
        memory = develop_memory_for_dt_action_blocker(env)
        action_blocker = DTActionBlocker(env.observation_space, model_name=action_blocking_model_name,
                                         memory=memory, penalty=min_penalty)
        agent.action_blocker = action_blocker
    else:
        agent = HeuristicWithTD(heuristic_func, use_model_only, algorithm_type, is_double, gamma,
                                env.action_space, env.observation_space.shape, mem_size, batch_size,
                                network_args, optimizer_type, policy_type, policy_args, replace, optimizer_args,
                                enable_action_blocking, min_penalty, goal, add_conservative_loss, alpha, **args)
    return run(env, n_games, agent, learning_type)


def run_with_dueling_td(env, n_games, learning_type, heuristic_func, use_model_only,
                        algorithm_type, is_double, gamma, mem_size, batch_size, network_args,
                        optimizer_type, policy_type, policy_args={},
                        replace=1000, optimizer_args={}, enable_action_blocking=False,
                        min_penalty=0, goal=None, add_conservative_loss=False, alpha=0.001,
                        use_ml_for_action_blocking=False, action_blocking_model_name=None, **args):
    if use_ml_for_action_blocking:
        agent = HeuristicWithDuelingTD(heuristic_func, use_model_only, algorithm_type, is_double, gamma,
                                       env.action_space, env.observation_space.shape, mem_size, batch_size,
                                       network_args, optimizer_type, policy_type, policy_args, replace, optimizer_args,
                                       False, 0, goal, add_conservative_loss, alpha, **args)
        memory = develop_memory_for_dt_action_blocker(env)
        action_blocker = DTActionBlocker(env.observation_space, model_name=action_blocking_model_name,
                                         memory=memory, penalty=min_penalty)
        agent.action_blocker = action_blocker
    else:
        agent = HeuristicWithDuelingTD(heuristic_func, use_model_only, algorithm_type, is_double, gamma,
                                       env.action_space, env.observation_space.shape, mem_size, batch_size,
                                       network_args, optimizer_type, policy_type, policy_args, replace, optimizer_args,
                                       enable_action_blocking, min_penalty, goal, add_conservative_loss, alpha, **args)
    return run(env, n_games, agent, learning_type)


def run_with_ddpg(env, n_games, learning_type, heuristic_func, use_model_only, tau, network_args,
                  actor_optimizer_type,
                  critic_optimizer_type, actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
                  max_size=1000000, batch_size=64, goal=None, **args):
    agent = HeuristicWithDDPG(heuristic_func, use_model_only, env.observation_space.shape, env.action_space, tau,
                              network_args, actor_optimizer_type,
                              critic_optimizer_type, actor_optimizer_args, critic_optimizer_args, gamma,
                              max_size, batch_size, goal, **args)
    return run(env, n_games, agent, learning_type)


def run_with_td3(env, n_games, learning_type, heuristic_func, use_model_only, tau, network_args,
                 actor_optimizer_type,
                 critic_optimizer_type, actor_optimizer_args={}, critic_optimizer_args={}, gamma=0.99,
                 max_size=1000000, batch_size=64, policy_update_interval=2, noise_std=0.2,
                 noise_clip=0.5, goal=None, **args):
    agent = HeuristicWithTD3(heuristic_func, use_model_only, env.observation_space.shape, env.action_space, tau,
                             network_args,
                             actor_optimizer_type, critic_optimizer_type, actor_optimizer_args, critic_optimizer_args,
                             gamma, max_size, batch_size, policy_update_interval, noise_std,
                             noise_clip, goal, **args)
    return run(env, n_games, agent, learning_type)


def run(env, n_games, agent, learning_type):
    learning_types = [LearningType.OFFLINE, LearningType.ONLINE] if learning_type == LearningType.BOTH else [
        learning_type]
    if type(n_games) == int:
        n_games_train = n_games
        n_games_test = n_games
    elif type(n_games) == tuple:
        n_games_train, n_games_test = n_games
    else:
        raise ValueError('n_games should either be int or tuple')

    scores_train = np.zeros((n_games_train * len(learning_types)))
    num_time_steps_train = 0
    for current_learning_type in learning_types:
        for i in range(n_games_train):
            score = 0
            observation = env.reset()
            done = False

            t = 0
            while not done and not have_we_ran_out_of_time(env, t):
                action = agent.get_action(env, learning_type, observation, True, t)

                observation_, reward, done, _ = env.step(action)

                if agent.enable_action_blocking and agent.initial_action_blocked and reward > 0:
                    reward *= -1
                score += reward

                agent.store_transition(observation, agent.get_original_action(env, learning_type,
                                                                              observation, True, t),
                                       reward, observation_, done)

                observation = observation_

                t += 1

            scores_train[i] = score
            num_time_steps_train += t
        agent.optimize(env, current_learning_type)
    if n_games_test == 0:
        if agent.enable_action_blocking:
            num_actions_blocked = agent.action_blocker.num_actions_blocked
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
        if agent.enable_action_blocking:
            num_actions_blocked_train = agent.action_blocker.num_actions_blocked
            agent.action_blocker.num_actions_blocked = 0
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
                action = agent.get_action(env, learning_type, observation, False)
                observation_, reward, done, _ = env.step(action)
                score += reward

                t += 1

            scores_test[i] = score
            num_time_steps_test += t

        if agent.enable_action_blocking:
            num_actions_blocked_test = agent.action_blocker.num_actions_blocked
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
