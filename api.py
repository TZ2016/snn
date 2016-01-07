from __future__ import division
import pprint
import pickle
import cgt
import copy
from cgt import nn
import numpy as np
from cgt.utility.param_collection import ParamCollection

import sfnn
import rnn
from utils.opt import *
from utils.debug import safe_path
from utils.utilities import NONE


# import traceback
# def _numpy_err_callback(type, flag):
#     print type, flag
#     traceback.print_stack()
#     raise FloatingPointError('refer to _numpy_err_callback for more details')
# np.seterr(divide='call', over='warn', invalid='call', under='warn')
# np.seterrcall(_numpy_err_callback)
# np.set_printoptions(precision=4, suppress=True)
cgt.check_source()  # this line will fail if CGT in use is not TZ2016's fork
print cgt.get_config(True)


def init(args):
    ws = {}
    ws['config'] = copy.deepcopy(args)
    _is_sto = any(_n != 0 for _n in args['num_sto'])
    _is_rec = any(_n != 0 for _n in args['num_mems'])
    assert not (_is_rec and _is_sto), "Stochastic recurrent units not supported"
    net_type = []
    if _is_sto: net_type.append('snn')
    else: net_type.append('dnn')
    if _is_rec: net_type.append('rnn')
    else: net_type.append('fnn')
    ws['type'] = net_type
    # TODO: add in the dbg_out
    if _is_rec:
        print "=========Start building a DRNN========="
        ws['f_train'] = rnn.step_once
        params, ws['f_step'], ws['f_loss'], ws['f_grad'], ws['f_init'], ws['f_surr'] = rnn.make_funcs(args)
    else:
        print "=========Start building a SFNN========="
        ws['f_train'] = sfnn.step_once
        params, ws['f_step'], ws['f_loss'], ws['f_grad'], ws['f_init'], ws['f_surr'] = sfnn.make_funcs(args)
    param_col = ParamCollection(params)
    ws['param_col'] = param_col
    _init_optim_state(ws)
    if ws['optim_state']['type'] == 'adam':
        ws['f_update'] = adam_update
    elif ws['optim_state']['type'] == 'rmsprop':
        ws['f_update'] = rmsprop_update
    else:
        raise ValueError
    param_col.set_value_flat(ws['optim_state']['theta'])
    print "Configurations"
    pprint.pprint(args)
    print "=========DONE BUILDING========="
    return ws


def _init_optim_state(ws, reset=False):
    if 'optim_state' in ws and not reset: return
    config = ws['config']
    if 'optim_state' in ws:
        print "Reusing cached optim_state"
        theta = ws['optim_state']['theta']
    elif 'snapshot' in config:
        print "Loading optim_state from previous snapshot: %s" % config['snapshot']
        ws['optim_state'] = pickle.load(open(config['snapshot'], 'r'))
        theta = ws['optim_state']['theta']
    else:
        init_method = config['init_theta']['distr']
        if init_method == 'XavierNormal':
            init_theta = nn.XavierNormal(**config['init_theta']['params'])
        elif init_method == 'gaussian':
            init_theta = nn.IIDGaussian(**config['init_theta']['params'])
        else:
            raise ValueError('unknown init distribution')
        theta = nn.init_array(init_theta, (ws['param_col'].get_total_size(), 1)).flatten()
    method = config['opt_method'].lower()
    if method == 'rmsprop':
        optim_create = lambda t: rmsprop_create(t, step_size=config['step_size'])
    elif method == 'adam':
        optim_create = lambda t: adam_create(t, step_size=config['step_size'])
    else:
        raise ValueError('unknown optimization method: %s' % method)
    if reset or 'optim_state' not in ws:
        ws['optim_state'] = optim_create(theta)


def _check(workspace, Xs=None, Ys=None, Ys_var=None, Ys_prec=None):
    config = workspace['config']
    # transform input if needed
    dX = Xs.shape[-1]
    assert dX == config['num_inputs']
    if Ys is None:
        assert Ys_var is None and Ys_prec is None
        Ys = Ys_var = Ys_prec = NONE
    else:
        dY = Ys.shape[-1]
        assert Ys_var is None or Ys_prec is None
        assert Ys_var is None, "for historical reasons"
        if Ys_prec is None and Ys_var is not None:
            # TODO_TZ: calculate the inverse for convenience
            'unreachable, todo in the future'
        assert dY is None or dY == config['num_outputs']
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
        Xs = np.expand_dims(Xs, axis=1)
        if Ys is not NONE:
            Ys = np.expand_dims(Ys, axis=1)
            Ys_prec = np.expand_dims(Ys_prec, axis=1)
    elif _ndim == 3:
        if 'fnn' in workspace['type'] and Xs.shape[1] > 1:
            Xs = np.reshape(Xs, (-1, 1, dX))
            if Ys is not NONE:
                Ys = np.reshape(Ys, (-1, 1, dY))
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


def forward(workspace, Xs,
            dbg_iter=None, dbg_done=None):
    config = workspace['config']
    assert 'rnn' not in workspace['type']
    Xs, Ys, Ys_prec = _check(workspace, Xs)
    N, T = Xs.shape[:2]
    B = config['size_batch']
    param_col = workspace['param_col']
    optim_state = workspace['optim_state']
    f_surr, f_step = workspace['f_surr'], workspace['f_step']
    if not config['debug']: dbg_iter = dbg_done = None
    for b in range(int(np.ceil(N / B))):
        _is = np.arange(b*B, min(N, B*(b+1)))
        _Xb, _Yb, _Yb_var = Xs[_is], Ys[_is], Ys_prec[_is]  # (B, T, dim)
        _xb = np.squeeze(_Xb, axis=1)
        out = f_step(_xb)[0]
        if dbg_iter: dbg_iter(-1, b*B, out, workspace)
    if dbg_done: dbg_done(workspace)
    return param_col, optim_state


def evaluate(workspace, Xs, Ys,
             Ys_var=None, Ys_prec=None,
             dbg_iter=None, dbg_done=None):
    config = workspace['config']
    assert 'rnn' not in workspace['type']
    Xs, Ys, Ys_prec = _check(workspace, Xs, Ys, Ys_var, Ys_prec)
    N, T = Xs.shape[:2]
    B = config['size_batch']
    param_col = workspace['param_col']
    optim_state = workspace['optim_state']
    f_surr, f_step = workspace['f_surr'], workspace['f_step']
    if not config['debug']: dbg_iter = dbg_done = None
    for b in range(int(np.ceil(N / B))):
        _is = np.arange(b*B, min(N, B*(b+1)))
        _Xb, _Yb, _Yb_var = Xs[_is], Ys[_is], Ys_prec[_is]  # (B, T, dim)
        _xb = np.squeeze(_Xb, axis=1)
        _yb = np.squeeze(_Yb, axis=1)
        _yb_prec = np.squeeze(_Yb_var, axis=1)
        out = f_surr(_xb, _yb_prec, _yb, num_samples=config['size_sample'])
        if dbg_iter: dbg_iter(-1, b*B, out, workspace)
    if dbg_done: dbg_done(workspace)
    return param_col, optim_state


def train(workspace, Xs, Ys,
          Ys_var=None, Ys_prec=None,
          dbg_iter=None, dbg_done=None):
    config = workspace['config']
    # pprint.pprint(config)
    # pprint.pprint(workspace)
    Xs, Ys, Ys_prec = _check(workspace, Xs, Ys, Ys_var, Ys_prec)
    # print "=========Start Training========="
    N, T = Xs.shape[:2]
    B = config['size_batch']
    M = config['rnn_steps']
    K = config['n_epochs']
    param_col = workspace['param_col']
    optim_state = workspace['optim_state']
    f_init = workspace['f_init']
    f_train, f_update = workspace['f_train'], workspace['f_update']
    f_surr, f_step = workspace['f_surr'], workspace['f_step']
    num_epochs, num_iters = -1, N
    if not config['debug']: dbg_iter = dbg_done = None
    # print "About to train for %d epochs" % K
    while num_epochs < K:
        if num_iters >= N:
            _ind = np.random.choice(N, replace=False, size=N)
            num_epochs, num_iters = num_epochs + 1, 0
            # print "Epoch %d starts" % num_epochs
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
    # print "=========DONE Training========="
    if 'dump_path' in config and config['dump_path']:
        save(config['dump_path'], workspace)
    if dbg_done: dbg_done(workspace)
    return param_col, optim_state


def save(path, ws):
    try:
        data = dict(type=ws['type'], optim_state=ws['optim_state'],
                    config=ws['config'], params_val=ws['param_col'].get_values())
        pickle.dump(data, safe_path(path, flag='w'))
        print "Snapshot successfully saved to %s" % (path, )
    except:
        print "Warning: saving params failed!"
        input("Save the params manually before too late")


if __name__ == "__main__":
    import os
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
