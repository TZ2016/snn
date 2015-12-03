from __future__ import division
import pprint
import pickle
import cgt
import os
from cgt import nn
import numpy as np
from cgt.utility.param_collection import ParamCollection

import sfnn
import rnn
from utils.opt import *


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
