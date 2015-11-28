from __future__ import division
import cgt
import time
import pprint
import pickle
from cgt import nn, utils
import numpy as np
from cgt.distributions import gaussian_diagonal
from cgt.utility.param_collection import ParamCollection
from layers import lstm_block, combo_layer
from opt import adam_create, adam_update, rmsprop_create, rmsprop_update
from debug import example_debug, safe_path


# TODO untested
def lstm_network_t(size_in, size_out, num_units, num_mems, dbg_out={}):
    def s_func_lstm(_in, _s_in, _s_out, name=''):
        c_prev = cgt.matrix(fixed_shape=(None, _s_out))
        h_prev = cgt.matrix(fixed_shape=(None, _s_out))
        c_cur, h_cur = lstm_block(h_prev, c_prev, _in, _s_in, _s_out, name)
        net_c_prev.append(c_prev)
        net_h_prev.append(h_prev)
        net_c_curr.append(c_cur)
        net_h_curr.append(h_cur)
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
    Xs = [cgt.matrix(fixed_shape=x.get_fixed_shape()) for _ in range(T)]
    C_0 = [cgt.matrix(fixed_shape=_c.get_fixed_shape()) for _c in c_in]
    H_0 = [cgt.matrix(fixed_shape=_h.get_fixed_shape()) for _h in h_in]
    loss, C_t, H_t, Ys = [], C_0, H_0, []
    for x in Xs:
        _out = f_lstm_t([x] + C_t + H_t)
        y, C_t, H_t = _out[0], _out[1:len(C_t)+1], _out[-len(H_t):]
        Ys.append(y)
    C_T, H_T = C_t, H_t
    params = f_lstm_t.get_parameters()
    return params, Xs, Ys, C_0, H_0, C_T, H_T


def make_funcs(config, dbg_out=None):
    params, Xs, Ys, C_0, H_0, C_T, H_T = lstm_network(
        config['T'], config['num_inputs'], config['num_outputs'],
        config['num_units'], config['num_mems']
    )

    size_batch = Xs[0].shape[0]
    net_inputs, net_outputs = Xs + C_0 + H_0, Ys + C_T + H_T
    f_step = cgt.function(net_inputs, net_outputs)

    assert isinstance(config['variance'], float)
    Ys_var = [cgt.fill(config['variance'], y.shape) for y in Ys]
    Ys_gt = [cgt.matrix(fixed_shape=y.get_fixed_shape()) for y in Ys]
    loss_vec = []
    for y_gt, y, y_var in zip(Ys_gt, Ys, Ys_var):
        _l = gaussian_diagonal.logprob(y_gt, y, y_var)
        loss_vec.append(_l)
    loss_vec = cgt.add_multi(loss_vec)
    if config['param_penal_wt'] > 0.:
        params_flat = cgt.concatenate([p.flatten() for p in params])
        loss_param = config['param_penal_wt'] * cgt.sum(params_flat ** 2)
        loss_vec += loss_param  # / size_batch
    loss = cgt.sum(loss_vec) / config['T'] / size_batch
    f_loss = cgt.function(net_inputs + Ys_var + Ys_gt, loss)

    grad = cgt.grad(loss, params)
    grad_flat = cgt.concatenate([g.flatten() for g in grad])
    f_grad = cgt.function(net_inputs + Ys_var + Ys_gt, grad_flat)

    return params, f_step, f_loss, f_grad, None, None


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
    while num_epochs < config['n_epochs']:
        ind = np.random.choice(X.shape[0], config['size_batch'])
        # ind = [num_iters]  # for ordered data
        x, y = X[ind], Y[ind]
        if config['variance'] == 'in':
            y_var = Y_var[ind]
            info = f_surr(x, y_var, y, num_samples=config['size_sample'])
        else:
            info = f_surr(x, y, num_samples=config['size_sample'])
        grad = info['grad']
        workspace['update'](param_col.flatten_values(grad), optim_state)
        param_col.set_value_flat(optim_state['theta'])
        num_iters += 1
        if num_iters == Y.shape[0]:
            num_epochs += 1
            num_iters = 0
            # TODO remove the below
            h_prob = np.exp(info['objective_unweighted'] - info['weights_raw_log'])
            print np.unique(np.round(h_prob, 2), return_counts=True)
            print np.unique(np.round(info['weights'], 3), return_counts=True)
            if num_epochs % 5 == 0:
                if config['variance'] == 'in':
                    _dbg = f_surr(X, Y_var, Y, num_samples=1, sample_only=True)
                else:
                    _dbg = f_surr(X, Y, num_samples=1, sample_only=True)
                pickle.dump(_dbg, safe_path('_sample_e%d.pkl' % num_epochs, out_path, 'w'))
        if dbg_iter:
            dbg_iter(num_epochs, num_iters, info, workspace)
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


def create(args):
    params, f_step, f_loss, f_grad, _, _ = make_funcs(args)
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
        # 'f_surr': f_surr,
        'f_step': f_step,
        'f_loss': f_loss,
        'f_grad': f_grad,
        'update': f_update,
    }
    print "Configurations"
    pprint.pprint(args)
    return workspace








class CGTSolver(object):
    def __init__(self, cgt_net,
                 learning_rate=1.0, rho=0.9):
        self.cgt_net = cgt_net
        theta = self.cgt_net.params_col.get_value_flat()
        self.optim_state = rmsprop_create(theta, learning_rate, rho)
        self.num_epochs = 0

    def step(self, num_step, train_input, train_output):
        input_patches = self.process_data(train_input)
        output_patches = self.process_data(train_output)
        for _ in range(num_step):
            self.step_once(input_patches, output_patches)

    def step_once(self, input_patches, output_patches):
        self.num_epochs += 1
        print "starting epoch", self.num_epochs
        tstart = time.time()
        losses = []
        cur_hiddens = self.cgt_net.init_hidden_layers()
        for (x, y) in zip(input_patches, output_patches):
            assert x.shape[0] == y.shape[0] == self.cgt_net.n_unroll
            assert x.shape[1] == self.cgt_net.size_input
            assert y.shape[1] == self.cgt_net.size_output
            out = self.cgt_net.f_loss_and_grad(x, y, *cur_hiddens)
            loss, grad, cur_hiddens = out[0], out[1], out[2:]
            rmsprop_update(grad, self.optim_state)
            self.cgt_net.params_col.set_value_flat(self.optim_state.theta)
            losses.append(loss)
        print "%.3f s/batch. avg loss = %.3f" % \
              ((time.time() - tstart) / len(losses), np.mean(losses))
        self.optim_state.step_size *= .98

    def process_data(self, data):
        """
        Return a list of numpy arrays of shape (n_unroll, size_batch, size_*)
        If data is used for input, size_* = size_input
        """
        assert data.ndim == 2
        N = data.shape[0]
        size_patch = self.cgt_net.n_unroll * self.cgt_net.size_batch
        num_patches = N // size_patch
        patches = np.split(data[:num_patches * size_patch], num_patches, axis=0)
        patches = [
            np.reshape(patch, (self.cgt_net.n_unroll, self.cgt_net.size_batch, -1))
            for patch in patches
        ]
        return patches

# def cgt_gru(size_input, size_mem, n_layers, size_output, size_batch):
#     inputs = [cgt.matrix() for i_layer in xrange(n_layers+1)]
#     outputs = []
#     for i_layer in xrange(n_layers):
#         prev_h = inputs[i_layer+1] # note that inputs[0] is the external input, so we add 1
#         x = inputs[0] if i_layer==0 else outputs[i_layer-1]
#         size_x = size_input if i_layer==0 else size_mem
#         update_gate = cgt.sigmoid(
#             nn.Affine(size_x, size_mem,name="i2u")(x)
#             + nn.Affine(size_mem, size_mem, name="h2u")(prev_h))
#         reset_gate = cgt.sigmoid(
#             nn.Affine(size_x, size_mem,name="i2r")(x)
#             + nn.Affine(size_mem, size_mem, name="h2r")(prev_h))
#         gated_hidden = reset_gate * prev_h
#         p2 = nn.Affine(size_mem, size_mem)(gated_hidden)
#         p1 = nn.Affine(size_x, size_mem)(x)
#         hidden_target = cgt.tanh(p1+p2)
#         next_h = (1.0-update_gate)*prev_h + update_gate*hidden_target
#         outputs.append(next_h)
#     category_activations = nn.Affine(size_mem, size_output,name="pred")(outputs[-1])
#     logprobs = nn.logsoftmax(category_activations)
#     outputs.append(logprobs)
#
#     return nn.Module(inputs, outputs)

if __name__ == "__main__":
    import yaml
    import time
    import os
    from data import scale_data, data_synthetic_a

    DUMP_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_tmp')
    PARAMS_PATH = os.path.join(DUMP_ROOT, '../sfnn_params.yaml')
    DEFAULT_ARGS = yaml.load(open(PARAMS_PATH, 'r'))
    DEFAULT_ARGS['dump_path'] = os.path.join(DUMP_ROOT,'_%d/' % int(time.time()))
    print "Default args:"
    pprint.pprint(DEFAULT_ARGS)

    X, Y, Y_var = data_synthetic_a(1000)
    X, Y, Y_var = scale_data(X, Y, Y_var=Y_var)
    problem = create(DEFAULT_ARGS)
    step(X, Y, problem, DEFAULT_ARGS, Y_var=Y_var)
