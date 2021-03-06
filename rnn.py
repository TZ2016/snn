from __future__ import division
import pickle
import cgt
import os
from cgt import nn
import numpy as np
import cgt.distributions as dist

from layers import lstm_block, combo_layer, s_func_ip
from utils import safe_io, safe_logadd


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
            s_funcs=(s_func_lstm, s_func_ip),
            o_funcs=(None, cgt.sigmoid),
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
    assert T > 0
    x, y, c_in, h_in, c_out, h_out = lstm_network_t(
        size_in, size_out, num_units, num_mems, dbg_out
    )
    f_lstm_t = nn.Module([x] + c_in + h_in, [y] + c_out + h_out)
    Xs = [cgt.matrix(fixed_shape=x.get_fixed_shape(), name="X%d"%t)
          for t in range(T)]
    C_0 = [cgt.matrix(fixed_shape=_c.get_fixed_shape()) for _c in c_in]
    H_0 = [cgt.matrix(fixed_shape=_h.get_fixed_shape()) for _h in h_in]
    loss, C_t, H_t, Ys = [], C_0, H_0, []
    for t, x in enumerate(Xs):
        _out = f_lstm_t([x] + C_t + H_t)
        y, C_t, H_t = _out[0], _out[1:len(C_t)+1], _out[1+len(C_t):]
        Ys.append(y)
        if t == 0:  C_1, H_1 = C_t, H_t
    C_T, H_T = C_t, H_t
    params = f_lstm_t.get_parameters()
    return params, Xs, Ys, C_0, H_0, C_T, H_T, C_1, H_1


def make_funcs(config, dbg_out=None):
    params, Xs, Ys, C_0, H_0, C_T, H_T, C_1, H_1 = lstm_network(
        config['rnn_steps'], config['num_inputs'], config['num_outputs'],
        config['num_units'], config['num_mems']
    )

    # basic
    size_batch = Xs[0].shape[0]
    dY = Ys[0].shape[-1]
    Ys_gt = [cgt.matrix(fixed_shape=(size_batch, dY), name='Y%d'%t)
             for t in range(len(Ys))]
    Ys_var = [cgt.tensor3(fixed_shape=(size_batch, dY, dY)) for _ in Ys]
    net_inputs, net_outputs = Xs + C_0 + H_0 + Ys_var, Ys + C_T + H_T

    # calculate loss
    loss_vec = []
    for i in range(len(Ys)):
        #     if i == 0: continue
        _l = dist.gaussian.logprob(Ys_gt[i], Ys[i], Ys_var[i])
        loss_vec.append(_l)
    loss_vec = cgt.add_multi(loss_vec)
    if config['weight_decay'] > 0.:
        params_flat = cgt.concatenate([p.flatten() for p in params])
        loss_param = config['weight_decay'] * cgt.sum(params_flat ** 2)
        loss_vec -= loss_param  # / size_batch
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
    f_step = cgt.function([Xs[0]] + C_0 + H_0, [Ys[0]] + C_1 + H_1)
    f_loss = cgt.function(net_inputs + Ys_gt, loss)
    f_grad = cgt.function(net_inputs + Ys_gt, grad)
    f_surr = cgt.function(net_inputs + Ys_gt, [loss] + net_outputs + grad)
    return params, f_step, f_loss, f_grad, f_init, f_surr


def step_once(param_col, optim_state, _Xb, _Yb, _Yb_var,
              f_update, f_surr, f_init, M, config={}):
    # all data params are of shape (batch_size, trajetory length, dim)
    B, T = _Xb.shape[:2]
    t, _Yb_hat = 0, []
    c_t, h_t = f_init(B)
    infos = []
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
        f_update(param_col.flatten_values(grad), optim_state)
        param_col.set_value_flat(optim_state['theta'])
        _Yb_hat.extend(ys_hat)
        infos.append(info)
    _Yb_hat = np.array(_Yb_hat).transpose(1, 0, 2)
    return infos, _Yb_hat


def step_tmp(param_col, optim_state, _Xb, _Yb, _Yb_var,
             f_update, f_surr, f_init, M, config={}):
    B, T = _Xb.shape[:2]
    t, _Yb_hat = 0, []
    # c_t, h_t = f_init(B)
    infos = []
    _best_h = []
    H = [0, 1]
    prev_ttl_loss = 0.
    for _b in range(B):
        loss_b, grad_b = [[], []], [[], []]
        for _t in range(T):
            _x, _y, _y_var = _Xb[_b, [_t]], _Yb[_b, [_t]], _Yb_var[_b, [_t]]
            for _h in range(2):
                # info = f_surr(*([np.c_[H[_h], _x]] + c_t + h_t + [_y_var] + [_y]))
                # loss, ys_hat, c_t, h_t, grad = info[0], \
                #                                info[1:1+M], \
                #                                info[1+M:1+M+len(c_t)], \
                #                                info[1+M+len(c_t):1+M+2*len(c_t)], \
                #                                info[1+M+2*len(c_t):]
                info = f_surr(np.c_[H[_h], _x], _y_var, _y)
                loss = info['objective']
                grad = info['grad']
                loss_b[_h].append(loss)
                grad_b[_h].append(grad)
        loss_b = np.array(loss_b).sum(axis=1)
        # print loss_b
        prev_ttl_loss += safe_logadd(loss_b)
        loss_b_n = loss_b - min(loss_b)
        _best_h.append(loss_b_n[0] - loss_b_n[1])
        for _h in range(2):
            for grad in grad_b[_h]:
                f_update(
                    (np.exp(loss_b_n[_h]) / np.sum(np.exp(loss_b_n))) *
                    param_col.flatten_values(grad), optim_state
                )
                param_col.set_value_flat(optim_state['theta'])
        # _Yb_hat.extend(ys_hat)
        # infos.append(info)
    # _Yb_hat = np.array(_Yb_hat).transpose(1, 0, 2)
    print _best_h
    print prev_ttl_loss
    return {
        'prev_ttl_loss': prev_ttl_loss
    }


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
            workspace['f_update'](param_col.flatten_values(grad), optim_state)
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
    safe_io(lambda f: pickle.dump(param_col.get_values(), f),
            'params.pkl', out_path, flag='w')
    safe_io(lambda f: pickle.dump(optim_state, f),
            '__snapshot.pkl', out_path, flag='w')
    return param_col, optim_state
