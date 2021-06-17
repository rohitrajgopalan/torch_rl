import numpy as np

from .agent import Agent
from torch_rl.action_blocker.action_blocker import ActionBlocker
from torch_rl.utils.utils import have_we_ran_out_of_time, get_next_discrete_action


def run(env, n_games, gamma, fc_dims, optimizer_type, optimizer_args={}, enable_action_blocking=False,
        min_penalty=0):
    agent = Agent(gamma, env.action_space.n, env.observation_space.shape,
                  fc_dims, optimizer_type, optimizer_args)

    if type(n_games) == int:
        n_games_train = n_games
        n_games_test = n_games
    elif type(n_games) == tuple:
        n_games_train, n_games_test = n_games
    else:
        raise ValueError('n_games should either be int or tuple')

    action_blocker = None
    if enable_action_blocking:
        action_blocker = ActionBlocker(input_dims=env.observation_space.shape, action_space=env.action_space,
                                       fc_dims=fc_dims, optimizer_type=optimizer_type, optimizer_args=optimizer_args,
                                       min_penalty=min_penalty)

    scores_train = np.zeros(n_games_train)

    num_time_steps_train = 0

    for i in range(n_games_train):
        score_train = 0
        observation = env.reset()
        done = False

        t = 0
        while not done and not have_we_ran_out_of_time(env, t):
            _, action_blocked, action = get_next_discrete_action(agent, env, observation,
                                                                 True, enable_action_blocking,
                                                                 action_blocker)

            if action is None:
                continue

            observation_, reward, done, _ = env.step(action)

            if enable_action_blocking and action_blocked:
                reward *= -1

            score_train += reward

            agent.learn(observation, reward, observation_, done)
            if enable_action_blocking:
                action_blocker.learn()

            observation = observation_

            t += 1

        scores_train[i] = score_train
        num_time_steps_train += t

    if n_games_test == 0:
        if enable_action_blocking:
            precision, recall = action_blocker.get_precision_and_recall()
        else:
            precision, recall = -1, -1

        return {
            'num_time_steps_train': num_time_steps_train,
            'avg_score_train': np.mean(scores_train),
            'action_blocker_precision_train': precision,
            'action_blocker_recall_train': recall,
            'num_actions_blocked_train': action_blocker.num_actions_blocked if enable_action_blocking else 0,
            'num_time_steps_test': 0,
            'avg_score_test': -1,
            'action_blocker_precision_test': -1,
            'action_blocker_recall_test': -1,
            'num_actions_blocked_test': 0,
            'avg_actor_loss': np.mean(agent.actor_loss_history),
            'avg_critic_loss': np.mean(agent.critic_loss_history)
        }
    else:
        if enable_action_blocking:
            train_precision, train_recall = action_blocker.get_precision_and_recall()
            num_actions_blocked_train = action_blocker.num_actions_blocked
            action_blocker.reset_confusion_matrix()
        else:
            train_precision, train_recall = -1, -1
            num_actions_blocked_train = 0

        scores_test = np.zeros(n_games_test)
        num_time_steps_test = 0

        for i in range(n_games_test):
            score_test = 0
            observation = env.reset()
            done = False

            t = 0
            while not done and have_we_ran_out_of_time(env, t):
                _, action_blocked, action = get_next_discrete_action(agent, env, observation,
                                                                     False, enable_action_blocking,
                                                                     action_blocker)

                if action is None:
                    continue

                observation_, reward, done, _ = env.step(action)

                if enable_action_blocking and action_blocked:
                    reward *= -1
                score_test += reward

                t += 1

            scores_test[i] = score_test
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
            'action_blocker_precision_train': train_precision,
            'action_blocker_recall_train': train_recall,
            'num_actions_blocked_train': num_actions_blocked_train,
            'num_time_steps_test': num_time_steps_test,
            'avg_score_test': np.mean(scores_test),
            'action_blocker_precision_test': test_precision,
            'action_blocker_recall_test': test_recall,
            'num_actions_blocked_test': num_actions_blocked_test,
            'actor_loss_history': np.mean(agent.actor_loss_history),
            'critic_loss_history': np.mean(agent.critic_loss_history)
        }
