# Learning Stochastic Feedforward Neural Networks

import os
import pickle
import cgt
from cgt.core import get_surrogate_func
from cgt import nn
import numpy as np
from cgt.distributions import gaussian_diagonal

from utils.debug import example_debug, safe_path
from layers import combo_layer


def hybrid_network(size_in, size_out, num_units, num_stos, dbg_out={}):
    assert len(num_units) == len(num_stos)
    net_in = cgt.matrix("X", fixed_shape=(None, size_in))
    prev_num_units, prev_out = size_in, net_in
    dbg_out['NET~in'] = net_in
    curr_layer = 1
    for (curr_num_units, curr_num_sto) in zip(num_units, num_stos):
        assert curr_num_units >= curr_num_sto >= 0
        prev_out = combo_layer(prev_out, prev_num_units, curr_num_units,
                               (curr_num_sto,),
                               o_funcs=(cgt.bernoulli, None),
                               name=str(curr_layer), dbg_out=dbg_out)
        dbg_out['L%d~out' % curr_layer] = prev_out
        prev_num_units = curr_num_units
        curr_layer += 1
    net_out = nn.Affine(prev_num_units, size_out,
                        name="InnerProd(%d->%d)" % (prev_num_units, size_out)
                        )(prev_out)
    dbg_out['NET~out'] = net_out
    return net_in, net_out


def make_funcs(config, dbg_out={}):
    net_in, net_out = hybrid_network(config['num_inputs'], config['num_outputs'],
                                     config['num_units'], config['num_sto'],
                                     dbg_out=dbg_out)
    if not config['dbg_out_full']: dbg_out = {}
    # def f_sample(_inputs, num_samples=1, flatten=False):
    #     _mean, _var = f_step(_inputs)
    #     _samples = []
    #     for _m, _v in zip(_mean, _var):
    #         _s = np.random.multivariate_normal(_m, np.diag(np.sqrt(_v)), num_samples)
    #         if flatten: _samples.extend(_s)
    #         else: _samples.append(_s)
    #     return np.array(_samples)
    Y = cgt.matrix("Y")
    params = nn.get_parameters(net_out)
    size_batch, size_out = net_out.shape
    inputs = [net_in]
    if config['no_bias']:
        print "Excluding bias"
        params = [p for p in params if not p.name.endswith(".b")]
    if config['variance'] == 'in':
        print "Input includes diagonal variance"
        # TODO_TZ diagonal for now
        in_var = cgt.matrix('V', fixed_shape=(None, config['num_inputs']))
        inputs.append(in_var)
        out_mean, out_var = net_out, in_var
    elif config['variance'] == 'out':  # net outputs variance
        print "Network outputs diagonal variance"
        cutoff = size_out // 2
        out_mean, out_var = net_out[:, :cutoff], net_out[:, cutoff:]
        # out_var = out_var ** 2 + 1.e-6
        out_var = cgt.exp(out_var) + 1.e-6
    else:
        print "Constant variance"
        # TODO make variance always the input
        assert isinstance(config['variance'], float)
        out_mean = net_out
        out_var = cgt.fill(config['variance'], [size_batch, size_out])
    net_out = [out_mean, out_var]
    loss_raw = gaussian_diagonal.logprob(Y, out_mean, out_var)
    if config['param_penal_wt'] != 0.:
        print "Applying penalty on parameter norm"
        assert config['param_penal_wt'] > 0
        params_flat = cgt.concatenate([p.flatten() for p in params])
        loss_param = config['param_penal_wt'] * cgt.sum(params_flat ** 2)
        loss_raw -= loss_param / size_batch
    # end of loss definition
    f_step = cgt.function(inputs, net_out)
    f_surr = get_surrogate_func(inputs + [Y], net_out,
                                [loss_raw], params, _dbg_out=dbg_out)

    # TODO_TZ f_step seems not to fail if X has wrong dim
    return params, f_step, None, None, None, f_surr


def step_once(param_col, optim_state, _Xb, _Yb, _Yb_var,
              f_update, f_surr, f_init, M):
    # all data params are of shape (batch_size, 1, dim)

    raise NotImplementedError


def step(X, Y, workspace, config, Y_var=None, dbg_iter=None, dbg_done=None):
    if config['debug'] and (dbg_iter is None or dbg_done is None):
        dbg_iter, dbg_done = example_debug(config, X, Y, Y_var=Y_var)
    if config['variance'] == 'in': assert Y_var is not None
    f_surr, f_step = workspace['f_surr'], workspace['f_step']
    param_col = workspace['param_col']
    optim_state = workspace['optim_state']
    num_epochs = num_iters = 0
    out_path = config['dump_path']
    print "Dump path: %s" % out_path
    # _dbg_infos = []
    while num_epochs < config['n_epochs']:
        ind = np.random.choice(X.shape[0], config['size_batch'])
        # ind = [num_iters]  # for ordered data
        x, y = X[ind], Y[ind]
        if config['variance'] == 'in':
            y_var = Y_var[ind]
            info = f_surr(x, y_var, y, num_samples=config['size_sample'])
        else:
            info = f_surr(x, y, num_samples=config['size_sample'])
        # _dbg_infos.append(info)
        grad = info['grad']
        workspace['update'](param_col.flatten_values(grad), optim_state)
        param_col.set_value_flat(optim_state['theta'])
        num_iters += 1
        if num_iters == Y.shape[0]:
            num_epochs += 1
            num_iters = 0
            # TODO remove the below
            # h_prob = np.exp(info['objective_unweighted'] - info['weights_raw_log'])
            # print np.unique(np.round(h_prob, 2), return_counts=True)
            # print np.unique(np.round(info['weights'], 3), return_counts=True)
            # if num_epochs % 5 == 0:
            #     if config['variance'] == 'in':
            #         _dbg = f_surr(X, Y_var, Y, num_samples=1, sample_only=True)
            #     else:
            #         _dbg = f_surr(X, Y, num_samples=1, sample_only=True)
            #     pickle.dump(_dbg, safe_path('_sample_e%d.pkl' % num_epochs, out_path, 'w'))
        if dbg_iter:
            dbg_iter(num_epochs, num_iters, info, workspace)
    # #######################
    # plt stochastic hidden activations
    # _pb = np.squeeze([np.exp(_i['objective_unweighted'] - _i['weights_raw_log'])
    #        for _i in _dbg_infos])
    # _pb_c = [np.unique(_p, return_counts=True)
    #          for _p in _pb]
    # p = np.array([_i[0] for _i in _pb_c])
    # c = np.array([_i[1] for _i in _pb_c])
    # import matplotlib.pyplot as plt
    # plt.bar(range(60), c[:,1])
    # plt.bar(range(60), c[:,2], bottom=c[:,1],color='r')
    # plt.bar(range(60), c[:,3], bottom=c[:, 2]+c[:,1],color='y')
    # plt.bar(range(60), c[:,0], bottom=c[:, 2]+c[:,1]+c[:,3],color='g')
    # pickle.dump([c, p], open('./asdf.pkl', 'w'))
    ###########################goi#########
    # save params
    out_path = config['dump_path']
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print "Saving params to %s" % out_path
    # pickle.dump(args, open(_safe_path('args.pkl'), 'w'))
    pickle.dump(param_col.get_values(), safe_path('params.pkl', out_path, 'w'))
    pickle.dump(optim_state, safe_path('__snapshot.pkl', out_path, 'w'))
    if dbg_done: dbg_done(workspace)
    return param_col, optim_state
