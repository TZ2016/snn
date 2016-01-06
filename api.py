from __future__ import division
import pprint
import pickle
import cgt
import copy
import traceback
import os
from cgt import nn
import numpy as np
from cgt.utility.param_collection import ParamCollection

import sfnn
import rnn
from utils.opt import *
from utils.debug import safe_path


def _numpy_err_callback(type, flag):
    print type, flag
    traceback.print_stack()
    raise FloatingPointError('refer to _numpy_err_callback for more details')
np.seterr(divide='call', over='warn', invalid='call', under='warn')
np.seterrcall(_numpy_err_callback)
np.set_printoptions(precision=4, suppress=True)
cgt.check_source()
print cgt.get_config(True)


def init(args):
    _is_sto = any(_n != 0 for _n in args['num_sto'])
    _is_rec = any(_n != 0 for _n in args['num_mems'])
    assert not (_is_rec and _is_sto), "Stochastic recurrent units not supported"
    net_type = []
    if _is_sto: net_type.append('snn')
    else: net_type.append('dnn')
    if _is_rec: net_type.append('rnn')
    else: net_type.append('fnn')
    # TODO: add in the dbg_out
    if _is_rec:
        print "=========Start building a DRNN========="
        f_train = rnn.step_once
        params, f_step, f_loss, f_grad, f_init, f_surr = rnn.make_funcs(args)
    else:
        print "=========Start building a SFNN========="
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
        'train': f_train,
        'config': copy.deepcopy(args),
    }
    print "Configurations"
    pprint.pprint(args)
    print "=========DONE BUILDING========="
    return workspace


def _check(Xs, Ys, workspace, Ys_var, Ys_prec):
    assert Ys_var is None or Ys_prec is None
    assert Ys_var is None, "for historical reasons"
    if Ys_prec is None and Ys_var is not None:
        # TODO_TZ: calculate the inverse for convenience
        'unreachable, todo in the future'
    config = workspace['config']
    # transform input if needed
    dX, dY = Xs.shape[-1], Ys.shape[-1]
    assert dX == config['num_inputs'] and dY == config['num_outputs']
    assert Xs.ndim == Ys.ndim and \
           Xs.shape[:-1] == Ys.shape[:-1], "X and Y must be compatible"
    if config['variance'] == 'in':
        assert Ys_prec is not None, "Y precision is required"
        assert Ys_prec.shape[:-1] == Ys.shape and Ys_prec.shape[-1] == dY
    else:
        Ys_prec = np.zeros(Ys.shape + (dY,))
        for i in np.ndindex(Ys_prec.shape[:-2]):
            Ys_prec[i] = np.identity(dY) / config['variance']
    _ndim = Xs.ndim
    if _ndim == 2:
        assert 'rnn' not in workspace['type']
        Xs, Ys = np.expand_dims(Xs, axis=1), np.expand_dims(Ys, axis=1)
        Ys_prec = np.expand_dims(Ys_prec, axis=1)
    elif _ndim == 3:
        # pass
        if 'fnn' in workspace['type'] and Xs.shape[1] > 1:
            Xs, Ys = np.reshape(Xs, (-1, 1, dX)), np.reshape(Ys, (-1, 1, dY))
            Ys_prec = np.reshape(Ys_prec, (-1, 1, dY))
    # various checks
    N, T = Xs.shape[:2]
    B = config['size_batch']
    M = config['rnn_steps']
    assert B <= N, "batch size too large"
    if 'fnn' in workspace['type']:
        assert M == 1, "no point to unroll a FNN"
        assert T == 1, "for FNN, T = 1; use RNN if T > 1"
    if 'snn' in workspace['type']:
        assert B == 1, "not yet supported"
    if 'dnn' in workspace['type']:
        assert config['size_sample'] == 1
    if 'rnn' in workspace['type']:
        assert (T / M) * M == T >= M, "T must be a multiple of M"
    return Xs, Ys, Ys_prec


def forward(workspace, Xs, Ys,
            Ys_var=None, Ys_prec=None,
            dbg_iter=None, dbg_done=None):
    config = workspace['config']
    pprint.pprint(config)
    pprint.pprint(workspace)
    Xs, Ys, Ys_prec = _check(workspace, Xs, Ys, Ys_var, Ys_prec)
    N, T = Xs.shape[:2]
    B = config['size_batch']
    M = config['rnn_steps']
    param_col = workspace['param_col']
    optim_state = workspace['optim_state']
    f_init = workspace['f_init']
    f_train, f_update = workspace['train'], workspace['update']
    f_surr, f_step = workspace['f_surr'], workspace['f_step']
    if not config['debug']: dbg_iter = dbg_done = None
    for b in range(int(np.ceil(N / B))):
        _is = np.arange(b*B, min(N, B*(b+1)))
        _Xb, _Yb, _Yb_var = Xs[_is], Ys[_is], Ys_prec[_is]  # (B, T, dim)
        dbg_data = f_train(param_col, optim_state, _Xb, _Yb, _Yb_var,
                           f_update, f_surr, f_init, M, config=config, no_update=True)
        if dbg_iter: dbg_iter(-1, b*B, dbg_data, workspace)
    if dbg_done: dbg_done(workspace)
    return param_col, optim_state


def train(workspace, Xs, Ys,
          Ys_var=None, Ys_prec=None,
          dbg_iter=None, dbg_done=None):
    config = workspace['config']
    pprint.pprint(config)
    pprint.pprint(workspace)
    Xs, Ys, Ys_prec = _check(workspace, Xs, Ys, Ys_var, Ys_prec)
    print "=========Start Training========="
    N, T = Xs.shape[:2]
    B = config['size_batch']
    M = config['rnn_steps']
    K = config['n_epochs']
    param_col = workspace['param_col']
    optim_state = workspace['optim_state']
    f_init = workspace['f_init']
    f_train, f_update = workspace['train'], workspace['update']
    f_surr, f_step = workspace['f_surr'], workspace['f_step']
    num_epochs, num_iters = -1, N
    if not config['debug']: dbg_iter = dbg_done = None
    print "About to train for %d epochs" % K
    while num_epochs < K:
        if num_iters >= N:
            _ind = np.random.choice(N, replace=False, size=N)
            num_epochs, num_iters = num_epochs + 1, 0
            print "Epoch %d starts" % num_epochs
            # import matplotlib.pyplot as plt
            # _b, _d = 0, 0  # which batch/dim to plot
            # plt.scatter(range(_Xb[_b,:,_d].size), _Yb[_b,:,_d])
            # plt.scatter(range(_Xb[_b,:,_d].size), _Xb[_b,:,_d], color='y')
            # plt.scatter(range(_Xb[_b,:,_d].size), np.array(_Yb_hat)[_b,:,_d], color='r')
            # plt.close()
        _is = _ind[num_iters:num_iters+B]
        _Xb, _Yb, _Yb_var = Xs[_is], Ys[_is], Ys_prec[_is]  # (B, T, dim)
        dbg_data = f_train(param_col, optim_state, _Xb, _Yb, _Yb_var,
                           f_update, f_surr, f_init, M, config=config)
        if dbg_iter: dbg_iter(num_epochs, num_iters, dbg_data, workspace)
        num_iters += B
    print "=========DONE Training========="
    if 'dump_path' in config and config['dump_path']:
        save(config['dump_path'], workspace)
    if dbg_done: dbg_done(workspace)
    return param_col, optim_state


def save(root_dir, ws):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    print "Saving params to %s" % (root_dir, )
    try:
        pickle.dump(ws['config'], safe_path('args.pkl', root_dir, 'w'))
        pickle.dump(ws['param_col'].get_values(), safe_path('params.pkl', root_dir, 'w'))
        pickle.dump(ws['optim_state'], safe_path('__snapshot.pkl', root_dir, 'w'))
    except:
        print "Warning: saving params failed!"
        input("Save the params manually before too late")


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
    # Xs, Ys = data_add(10, 50, dim=2)

    # feed-forward datset
    X, Y, Y_var = data_synthetic_a(1000)
    X, Y, Y_var = scale_data(X, Y, Y_var=Y_var)
    Xs = np.expand_dims(X, axis=1)
    Ys = np.expand_dims(Y, axis=1)
    Ys_var = np.expand_dims(Y_var, axis=1)

    DEFAULT_ARGS.update({
        # test RNN
        # 'num_inputs': 2,
        # 'num_outputs': 2,
        # 'num_units': [6],
        # 'num_sto': [0],  # not used
        # 'variance': 0.001,
        # 'size_sample': 1,
        # 'num_mems': [4],
        # 'rnn_steps': 5,

        # test SFNN
        'num_inputs': 1,
        'num_outputs': 1,
        'num_units': [4, 4, 2],
        'num_sto': [0, 2, 2],  # not used
        'variance': 0.05,
        'size_sample': 1,
        'num_mems': [0, 0, 0],
        'rnn_steps': 1,
    })
    print "Using arguments:"
    pprint.pprint(DEFAULT_ARGS)

    problem = init(DEFAULT_ARGS)
    train(problem, Xs, Ys, Ys_prec=None)
