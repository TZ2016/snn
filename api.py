from __future__ import division
import pprint
import pickle
import cgt
import traceback
import os
from cgt import nn
import numpy as np
from cgt.utility.param_collection import ParamCollection

import sfnn
import rnn
from utils.opt import *
from utils.debug import safe_path


def err_handler(type, flag):
    print type, flag
    traceback.print_stack()
    raise FloatingPointError('refer to err_handler for more details')
np.seterr(divide='call', over='warn', invalid='call', under='warn')
np.seterrcall(err_handler)
np.set_printoptions(precision=4, suppress=True)
print cgt.get_config(True)
cgt.check_source()


def create_net(args):
    _is_sto = any(_n != 0 for _n in args['num_sto'])
    _is_rec = any(_n != 0 for _n in args['num_mems'])
    assert not (_is_rec and _is_sto), "Stochastic recurrent units not supported"
    net_type = []
    # TODO: add in the dbg_out
    if _is_rec:
        print "=========Start building a DRNN========="
        net_type.extend(['rnn', 'dnn'])
        f_train = rnn.step_once
        params, f_step, f_loss, f_grad, f_init, f_surr = rnn.make_funcs(args)
    else:
        print "=========Start building a SFNN========="
        net_type.extend(['snn', 'sfnn', 'fnn'])
        f_train = sfnn.step_once
        params, f_step, f_loss, f_grad, f_init, f_surr = sfnn.make_funcs(args)
    param_col = ParamCollection(params)
    if 'snapshot' in args:
        print "Loading params from previous snapshot: %s" % args['snapshot']
        optim_state = pickle.load(open(args['snapshot'], 'r'))
        assert isinstance(optim_state, dict)
        if optim_state['type'] == 'adam':
            f_update = adam_update
        elif optim_state['type'] == 'rmsprop':
            f_update = rmsprop_update
        else:
            raise ValueError
    else:
        theta = param_col.get_value_flat()
        method = args['opt_method'].lower()
        print "Initializing theta from fresh"
        if method == 'rmsprop':
            optim_state = rmsprop_create(theta, step_size=args['step_size'])
            f_update = rmsprop_update
        elif method == 'adam':
            optim_state = adam_create(theta, step_size=args['step_size'])
            f_update = adam_update
        else:
            raise ValueError('unknown optimization method: %s' % method)
        init_method = args['init_theta']['distr']
        if init_method == 'XavierNormal':
            init_theta = nn.XavierNormal(**args['init_theta']['params'])
        elif init_method == 'gaussian':
            init_theta = nn.IIDGaussian(**args['init_theta']['params'])
        else:
            raise ValueError('unknown init distribution')
        optim_state['theta'] = nn.init_array(
            init_theta, (param_col.get_total_size(), 1)).flatten()
    param_col.set_value_flat(optim_state['theta'])
    # TODO: make workspace a proper class
    workspace = {
        'type': net_type,
        'optim_state': optim_state,
        'param_col': param_col,
        'f_surr': f_surr,
        'f_step': f_step,
        'f_loss': f_loss,
        'f_init': f_init,
        'f_grad': f_grad,
        'update': f_update,
        'train': f_train
    }
    print "Configurations"
    pprint.pprint(args)
    print "=========DONE BUILDING========="
    return workspace


def train(Xs, Ys, workspace, config, Ys_var=None):
    print "=========Start Training========="
    # perform input check
    assert Xs.ndim == Ys.ndim == 3, "must of shape (N, T, dX)"
    assert Xs.shape[:2] == Ys.shape[:2], "X and Y of same shape (N, T)"
    if config['variance'] == 'in':
        assert Ys_var is not None, "Y variance is required"
        assert Ys_var.shape == Ys.shape, "Y var of same shape as Y"
    else:
        Ys_var = config['variance'] * np.ones_like(Ys)
    (N, T, dX), dY = Xs.shape, Ys.shape[-1]
    assert dX == config['num_inputs'] and dY == config['num_outputs']
    B = config['size_batch']
    M = config['rnn_steps']
    K = config['n_epochs']
    assert B <= N, "batch size too large"
    if 'fnn' in workspace['type']:
        assert config['rnn_step'] == 1, "no point to unroll a FNN"
        assert T == 1, "for FNN, T = 1; use RNN if T > 1"
    if 'snn' in workspace['type']:
        assert B == 1, "not yet supported"
    if 'rnn' in workspace['type']:
        assert (T / M) * M == T >= M, "T must be a multiple of M"
    assert 'dump_path' in config and config['dump_path'], 'path required'
    out_path = config['dump_path']
    print "Dump path: %s" % out_path
    param_col = workspace['param_col']
    optim_state = workspace['optim_state']
    f_init = workspace['f_init']
    f_train, f_update = workspace['train'], workspace['update']
    f_surr, f_step = workspace['f_surr'], workspace['f_step']
    num_epochs = num_iters = 0
    print "About to train for %d epochs" % K
    while num_epochs < K:
        _is = np.random.choice(N, size=B)  # a list of B indices
        _Xb, _Yb, _Yb_var = Xs[_is], Ys[_is], Ys_var[_is]  # (B, T, dim)
        f_train(param_col, optim_state, _Xb, _Yb, _Yb_var,
                f_update, f_surr, f_init, M)
        num_iters += 1
        if num_iters == N:
            print "Epoch %d ends" % num_epochs
            num_epochs += 1
            num_iters = 0
            # import matplotlib.pyplot as plt
            # _b, _d = 0, 0  # which batch/dim to plot
            # plt.scatter(range(_Xb[_b,:,_d].size), _Yb[_b,:,_d])
            # plt.scatter(range(_Xb[_b,:,_d].size), _Xb[_b,:,_d], color='y')
            # plt.scatter(range(_Xb[_b,:,_d].size), np.array(_Yb_hat)[_b,:,_d], color='r')
            # plt.close()
    # save params
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print "Saving params"
    # pickle.dump(args, open(_safe_path('args.pkl'), 'w'))
    pickle.dump(param_col.get_values(), safe_path('params.pkl', out_path, 'w'))
    pickle.dump(optim_state, safe_path('__snapshot.pkl', out_path, 'w'))
    print "=========DONE Training========="
    return param_col, optim_state


if __name__ == "__main__":
    import yaml
    import time
    from utils.data import *

    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    DUMP_ROOT = os.path.join(CUR_DIR, '_tmp')
    PARAMS_PATH = os.path.join(CUR_DIR, 'default_params.yaml')
    DEFAULT_ARGS = yaml.load(open(PARAMS_PATH, 'r'))
    DEFAULT_ARGS['dump_path'] = os.path.join(DUMP_ROOT, '_%d/' % int(time.time()))

    # recurrent dataset
    Xs, Ys = data_add(10, 50, dim=2)

    # feed-forward datset
    # X, Y, Y_var = data_synthetic_a(1000)

    # X, Y, Y_var = scale_data(X, Y, Y_var=Y_var)

    DEFAULT_ARGS.update({
    })
    print "Using arguments:"
    pprint.pprint(DEFAULT_ARGS)

    problem = create_net(DEFAULT_ARGS)
    train(Xs, Ys, problem, DEFAULT_ARGS, Ys_var=None)
