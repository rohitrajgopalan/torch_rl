import torch.optim as optimizer

from .types import NetworkOptimizer


def get_torch_optimizer(params, optimizer_type, optimizer_args):
    if optimizer_type == NetworkOptimizer.ADAM:
        learning_rate = optimizer_args['learning_rate'] if 'learning_rate' in optimizer_args else 0.001
        beta_m = optimizer_args['beta_m'] if 'beta_m' in optimizer_args else 0.9
        beta_v = optimizer_args['beta_v'] if 'beta_v' in optimizer_args else 0.999
        weight_decay = optimizer_args['weight_decay'] if 'weight_decay' in optimizer_args else 0
        epsilon = optimizer_args['epsilon'] if 'epsilon' in optimizer_args else 1e-08

        return optimizer.Adam(params, lr=learning_rate, betas=(beta_m, beta_v), eps=epsilon, weight_decay=weight_decay)

    elif optimizer_type == NetworkOptimizer.ADAMAX:
        learning_rate = optimizer_args['learning_rate'] if 'learning_rate' in optimizer_args else 0.002
        beta_m = optimizer_args['beta_m'] if 'beta_m' in optimizer_args else 0.9
        beta_v = optimizer_args['beta_v'] if 'beta_v' in optimizer_args else 0.999
        weight_decay = optimizer_args['weight_decay'] if 'weight_decay' in optimizer_args else 0
        epsilon = optimizer_args['epsilon'] if 'epsilon' in optimizer_args else 1e-08

        return optimizer.Adamax(params, lr=learning_rate, betas=(beta_m, beta_v), eps=epsilon,
                                weight_decay=weight_decay)

    elif optimizer_type == NetworkOptimizer.ADAGRAD:
        learning_rate = optimizer_args['learning_rate'] if 'learning_rate' in optimizer_args else 0.01
        lr_decay = optimizer_args['lr_decay'] if 'lr_decay' in optimizer_args else 0
        weight_decay = optimizer_args['weight_decay'] if 'weight_decay' in optimizer_args else 0
        initial_accumulator_value = optimizer_args[
            'initial_accumulator_value'] if 'initial_accumulator_value' in optimizer_args else 0
        epsilon = optimizer_args['epsilon'] if 'epsilon' in optimizer_args else 1e-10

        return optimizer.Adagrad(params, lr=learning_rate, lr_decay=lr_decay, weight_decay=weight_decay,
                                 initial_accumulator_value=initial_accumulator_value,
                                 eps=epsilon)

    elif optimizer_type == NetworkOptimizer.RMSPROP:
        learning_rate = optimizer_args['learning_rate'] if 'learning_rate' in optimizer_args else 0.01
        weight_decay = optimizer_args['weight_decay'] if 'weight_decay' in optimizer_args else 0
        epsilon = optimizer_args['epsilon'] if 'epsilon' in optimizer_args else 1e-08
        alpha = optimizer_args['alpha'] if 'alpha' in optimizer_args else 0.99
        momentum = optimizer_args['momentum'] if 'momentum' in optimizer_args else 0
        centered = optimizer_args['centered'] if 'centered' in optimizer_args else False

        return optimizer.RMSprop(params, lr=learning_rate, weight_decay=weight_decay, eps=epsilon, alpha=alpha,
                                 momentum=momentum,
                                 centered=centered)

    elif optimizer_type == NetworkOptimizer.SGD:
        learning_rate = optimizer_args['learning_rate'] if 'learning_rate' in optimizer_args else 0.01
        momentum = optimizer_args['momentum'] if 'momentum' in optimizer_args else 0
        weight_decay = optimizer_args['weight_decay'] if 'weight_decay' in optimizer_args else 0
        dampening = optimizer_args['dampening'] if 'weight_decay' in optimizer_args else 0
        nesterov = optimizer_args['nesterov'] if 'nesterov' in optimizer_args else False

        return optimizer.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                             dampening=dampening, nesterov=nesterov)


def have_we_ran_out_of_time(env, current_t):
    return hasattr(env, '_max_episode_steps') and current_t == env._max_episode_steps


def get_hidden_layer_sizes(fc_dims):
    if type(fc_dims) == int:
        return fc_dims, fc_dims
    elif type(fc_dims) in [list, tuple]:
        return fc_dims[0], fc_dims[1]
    else:
        raise TypeError('fc_dims should be integer, list or tuple')
