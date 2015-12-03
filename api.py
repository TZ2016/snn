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
        print "=========Building a DRNN========="
        net_type.extend(['rnn', 'dnn'])
        params, f_step, f_loss, f_grad, f_init, f_surr = rnn.make_funcs(args)
    else:
        print "=========Building a SFNN========="
        net_type.extend(['snn', 'sfnn', 'fnn'])
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
    }
    print "Configurations"
    pprint.pprint(args)
    print "=========DONE BUILDING========="
    return workspace


if __name__ == "__main__":
    import yaml
    import time
    from utils.data import *

    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    DUMP_ROOT = os.path.join(CUR_DIR, '_tmp')
    PARAMS_PATH = os.path.join(CUR_DIR, 'default_params.yaml')
    DEFAULT_ARGS = yaml.load(open(PARAMS_PATH, 'r'))
    DEFAULT_ARGS['dump_path'] = os.path.join(DUMP_ROOT, '_%d/' % int(time.time()))
    print "Default args:"
    pprint.pprint(DEFAULT_ARGS)

    # recurrent dataset
    # Xs, Ys = data_add(10, 50, dim=2)

    # feed-forward datset
    X, Y, Y_var = data_synthetic_a(1000)

    X, Y, Y_var = scale_data(X, Y, Y_var=Y_var)

    # DEFAULT_ARGS.update({
    #
    # })
    problem = create_net(DEFAULT_ARGS)
    step(X, Y, problem, DEFAULT_ARGS, Y_var=Y_var)
