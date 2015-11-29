from __future__ import division
import pprint
import pickle
import cgt
import os
from cgt import nn
import numpy as np
from cgt.distributions import gaussian_diagonal
from cgt.utility.param_collection import ParamCollection

from layers import lstm_block, combo_layer
from utils.opt import *
from utils.debug import safe_path


def lstm_network_t(size_in, size_out, num_units, num_mems, dbg_out={}):
    def s_func_lstm(_in, _s_in, _s_out, name=''):
        c_prev = cgt.matrix(fixed_shape=(None, _s_out))
        h_prev = cgt.matrix(fixed_shape=(None, _s_out))
        c_cur, h_cur = lstm_block(h_prev, c_prev, _in, _s_in, _s_out, name)
        net_c_prev.append(c_prev)
        net_h_prev.append(h_prev)
        net_c_curr.append(c_cur)
        net_h_curr.append(h_cur)
        return h_cur
    assert len(num_units) == len(num_mems)
    net_c_prev, net_h_prev, net_c_curr, net_h_curr = [], [], [], []
    net_in = cgt.matrix(fixed_shape=(None, size_in))
    prev_num_units, prev_out = size_in, net_in
    curr_layer = 1
    for curr_num_units, curr_num_mem in zip(num_units, num_mems):
        assert curr_num_units >= curr_num_mem >= 0
        prev_out = combo_layer(
            prev_out, prev_num_units, curr_num_units,
            (curr_num_mem,),
            s_funcs=(s_func_lstm, None),
            name=str(curr_layer), dbg_out=dbg_out
        )
        dbg_out['L%d~out' % curr_layer] = prev_out
        prev_num_units = curr_num_units
        curr_layer += 1
    net_out = nn.Affine(prev_num_units, size_out,
                        name="Out")(prev_out)
    dbg_out['NET~out'] = net_out
    return net_in, net_out, net_c_prev, net_h_prev, net_c_curr, net_h_curr


def lstm_network(T, size_in, size_out, num_units, num_mems, dbg_out={}):
    x, y, c_in, h_in, c_out, h_out = lstm_network_t(
        size_in, size_out, num_units, num_mems, dbg_out
    )
    f_lstm_t = nn.Module([x] + c_in + h_in, [y] + c_out + h_out)
    Xs = [cgt.matrix(fixed_shape=x.get_fixed_shape(), name="X%d"%t)
          for t in range(T)]
    C_0 = [cgt.matrix(fixed_shape=_c.get_fixed_shape()) for _c in c_in]
    H_0 = [cgt.matrix(fixed_shape=_h.get_fixed_shape()) for _h in h_in]
    loss, C_t, H_t, Ys = [], C_0, H_0, []
    for x in Xs:
        _out = f_lstm_t([x] + C_t + H_t)
        y, C_t, H_t = _out[0], _out[1:len(C_t)+1], _out[1+len(C_t):]
        Ys.append(y)
    C_T, H_T = C_t, H_t
    params = f_lstm_t.get_parameters()
    return params, Xs, Ys, C_0, H_0, C_T, H_T


def make_funcs(config, dbg_out=None):
    params, Xs, Ys, C_0, H_0, C_T, H_T = lstm_network(
        config['rnn_steps'], config['num_inputs'], config['num_outputs'],
        config['num_units'], config['num_mems']
    )

    # basic
    size_batch = Xs[0].shape[0]
    net_inputs, net_outputs = Xs + C_0 + H_0, Ys + C_T + H_T
    Ys_gt = [cgt.matrix(fixed_shape=y.get_fixed_shape(), name='Y%d'%t)
             for t, y in enumerate(Ys)]
    Ys_var = [cgt.matrix(fixed_shape=y.get_fixed_shape()) for y in Ys]
    net_inputs += Ys_var  # mandatory variance as input

    # calculate loss
    loss_vec = []
    for i in range(len(Ys)):
        #     if i == 0: continue
        _l = gaussian_diagonal.logprob(Ys_gt[i], Ys[i], Ys_var[i])
        loss_vec.append(_l)
    loss_vec = cgt.add_multi(loss_vec)
    if config['param_penal_wt'] > 0.:
        params_flat = cgt.concatenate([p.flatten() for p in params])
        loss_param = config['param_penal_wt'] * cgt.sum(params_flat ** 2)
        loss_vec += loss_param  # / size_batch
    loss = cgt.sum(loss_vec) / config['rnn_steps'] / size_batch
    grad = cgt.grad(loss, params)

    # functions
    def f_init(size_batch):
        c_0, h_0 = [], []
        for _n_m in config['num_mems']:
            if _n_m > 0:
                c_0.append(np.zeros((size_batch, _n_m)))
                h_0.append(np.zeros((size_batch, _n_m)))
        return c_0, h_0
    f_step = cgt.function(net_inputs, net_outputs)
    f_loss = cgt.function(net_inputs + Ys_gt, loss)
    f_grad = cgt.function(net_inputs + Ys_gt, grad)
    f_surr = cgt.function(net_inputs + Ys_gt, [loss] + net_outputs + grad)
    return params, f_step, f_loss, f_grad, f_init, f_surr


def step(Xs, Ys, workspace, config, Ys_var=None):
    assert Xs.shape[:2] == Ys.shape[:2]
    (N, T, dX), dY = Xs.shape, Ys.shape[-1]
    M = config['rnn_steps']
    B = config['size_batch']
    assert B <= N
    assert (T / M) * M == T >= M
    assert dX == config['num_inputs'] and dY == config['num_outputs']
    if config['variance'] == 'in':
        assert Ys_var is not None and Ys_var.shape == Ys.shape
    else:
        Ys_var = config['variance'] * np.ones_like(Ys)
    f_init = workspace['f_init']
    f_surr, f_step = workspace['f_surr'], workspace['f_step']
    param_col = workspace['param_col']
    optim_state = workspace['optim_state']
    out_path = config['dump_path']
    print "Dump path: %s" % out_path
    num_epochs = num_iters = 0
    while num_epochs < config['n_epochs']:
        _is = np.random.choice(N, size=B)  # this is a list
        _Xb, _Yb, _Yb_var = Xs[_is], Ys[_is], Ys_var[_is]  # a batch of traj
        t, _Yb_hat = 0, []
        c_t, h_t = f_init(B)
        while t + M <= T:
            _xbs = list(_Xb[:, t:t+M].transpose(1, 0, 2))
            _ybs = list(_Yb[:, t:t+M].transpose(1, 0, 2))
            _ybs_var = list(_Yb_var[:, t:t+M].transpose(1, 0, 2))
            t += M
            info = f_surr(*(_xbs + c_t + h_t + _ybs_var + _ybs))
            loss, ys_hat, c_t, h_t, grad = info[0], \
                                           info[1:1+M], \
                                           info[1+M:1+M+len(c_t)], \
                                           info[1+M+len(c_t):1+M+2*len(c_t)], \
                                           info[1+M+2*len(c_t):]
            workspace['update'](param_col.flatten_values(grad), optim_state)
            param_col.set_value_flat(optim_state['theta'])
            _Yb_hat.extend(ys_hat)
        _Yb_hat = np.array(_Yb_hat).transpose(1, 0, 2)
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
    print "Saving params to %s" % out_path
    # pickle.dump(args, open(_safe_path('args.pkl'), 'w'))
    pickle.dump(param_col.get_values(), safe_path('params.pkl', out_path, 'w'))
    pickle.dump(optim_state, safe_path('__snapshot.pkl', out_path, 'w'))
    return param_col, optim_state


def create(args):
    assert all(_n == 0 for _n in args['num_sto']), "not supported"
    params, f_step, f_loss, f_grad, f_init, f_surr = make_funcs(args)
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
    workspace = {
        'optim_state': optim_state,
        'param_col': param_col,
        'f_surr': f_surr,
        'f_step': f_step,
        'f_loss': f_loss,
        'f_init': f_init,
        'f_grad': f_grad,
        'update': f_update,
    }
    print "Configurations"
    pprint.pprint(args)
    return workspace


if __name__ == "__main__":
    import yaml
    import time
    import os
    from utils.data import *

    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    DUMP_ROOT = os.path.join(CUR_DIR, '_tmp')
    PARAMS_PATH = os.path.join(CUR_DIR, 'default_params.yaml')
    DEFAULT_ARGS = yaml.load(open(PARAMS_PATH, 'r'))
    DEFAULT_ARGS['dump_path'] = os.path.join(DUMP_ROOT, '_%d/' % int(time.time()))
    print "Default args:"
    pprint.pprint(DEFAULT_ARGS)

    Xs, Ys = data_add(10, 50, dim=2)
    # Xs, Ys = data_add(10, 50, 2)
    DEFAULT_ARGS.update({
        'num_inputs': 2,
        'num_outputs': 2,
        'num_units': [6],
        'num_sto': [0],  # not used
        'variance': 0.001,
        'size_sample': 1,
        'num_mems': [4],
        'rnn_steps': 5,
    })
    problem = create(DEFAULT_ARGS)
    step(Xs, Ys, problem, DEFAULT_ARGS)
