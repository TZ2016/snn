import cgt
from cgt import nn


# def mux_layer(X, size_in, size_out, splits, funcs):
#     if not isinstance(funcs, (list, tuple)):
#         funcs = [funcs] * (len(splits) + 1)
#     assert len(splits) + 1 == len(funcs) > 0
#     splits.append(size_out)
#     assert all(splits[i] < splits[i+1] for i in xrange(len(splits) - 1))
#     curr, outs = 0, []
#     for _s, _f in zip(splits, funcs):
#         _o = _f(X, size_in, _s - curr)
#         outs.append(_o)
#         curr = _s
#     return outs
#
#
# def demux_layer(Xs, funcs):
#     assert isinstance(Xs, list)
#     assert len(funcs) == len(Xs)
#     curr, outs = 0, []
#     for _X, _f in zip(Xs, funcs):
#         _o = _f(_X) if _f is not None else _X
#         outs.append(_o)
#     out = cgt.concatenate(outs, axis=1)
#     return out


def mask_layer(func, X, size_in, i_start, i_end=None):
    if i_end is None:
        i_start, i_end = 0, i_start
    assert isinstance(i_start, int) and isinstance(i_end, int)
    assert -1 < i_start <= i_end <= size_in
    if i_end == i_start:
        return X
    if i_end - i_start == size_in:
        return func(X)
    outs = []
    if i_start > 0:
        outs.append(X[:, :i_start])
    outs.append(func(X[:, i_start:i_end]))
    if i_end < size_in:
        outs.append(X[:, i_end:])
    out = cgt.concatenate(outs, axis=1)
    return out


def combo_layer(X, size_in, size_out, splits,
                s_funcs=None, o_funcs=None,
                name='?', dbg_out={}):
    """
    Create a combination of specified sub-layers and non-linearity.

    :param X: symbolic input
    :param size_in: input size (except batch)
    :param size_out: output size (except batch)
    :param splits: split points for applying each sub-layer
    :param s_funcs: list of functions to create each sub-layer
        of signature (input, in_size, out_size, name) -> output
    :param o_funcs: list of non-linearity functions
        of signature (input) -> output
    :param name: layer name
    :type name: str
    :return: symbolic output

    Note for s_funcs and o_funcs:
        - broadcasting enabled if not a list
        - element with value None has no effect (skip)
    """

    assert isinstance(splits, tuple) and len(splits) > 0
    assert all(splits[i] < splits[i+1] for i in xrange(len(splits) - 1))
    assert splits[0] >= 0 and splits[-1] <= size_out
    splits = list(splits)
    assert not isinstance(o_funcs, list) and not isinstance(s_funcs, list)
    o_funcs = list(o_funcs) if isinstance(o_funcs, tuple) \
        else [o_funcs] * (len(splits) + 1)
    s_funcs = list(s_funcs) if isinstance(s_funcs, tuple) \
        else [s_funcs] * (len(splits) + 1)
    assert len(splits) + 1 == len(o_funcs) == len(s_funcs)
    if splits[0] == 0:
        splits.pop(0)
        o_funcs.pop(0)
        s_funcs.pop(0)
    if len(splits) > 0 and splits[-1] == size_out:
        splits.pop()
        o_funcs.pop()
        s_funcs.pop()
    curr, names, ins, outs = 0, [], [], []
    splits.append(size_out)
    for _split, _f_s, _f_o in zip(splits, s_funcs, o_funcs):
        _name = 'L%s[%d:%d]' % (name, curr, _split)
        _s_out = _split - curr
        _i = X if _f_s is None else _f_s(X, size_in, _s_out, name=_name)
        _o = _i if _f_o is None else _f_o(_i)
        curr = _split
        ins.append(_i)
        outs.append(_o)
        names.append(_name)
    out = cgt.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]
    dbg_out.update(dict(zip([_n + '~in' for _n in names], ins)))
    dbg_out.update(dict(zip([_n + '~out' for _n in names], outs)))
    return out


def s_func_ip(X, size_in, size_out, name):
    return nn.Affine(size_in, size_out, name=name)(X)


def lstm_block(h_prev, c_prev, x_curr, size_x, size_c, name=''):
    """
    Construct a LSTM cell block of specified number of cells

    :param h_prev: self activations at previous time step
    :param c_prev: self memory state at previous time step
    :param x_curr: inputs from previous layer at current time step
    :param size_x: size of inputs
    :param size_c: size of both c and h
    :return: c and h at current time step
    :rtype:
    """
    input_sums = nn.Affine(size_x, 4 * size_c, name=name+'*x')(x_curr) + \
                 nn.Affine(size_c, 4 * size_c, name=name+'*h')(h_prev)
    c_new = cgt.tanh(input_sums[:, 3*size_c:])
    sigmoid_chunk = cgt.sigmoid(input_sums[:, :3*size_c])
    in_gate = sigmoid_chunk[:, :size_c]
    forget_gate = sigmoid_chunk[:, size_c:2*size_c]
    out_gate = sigmoid_chunk[:, 2*size_c:3*size_c]
    c_curr = forget_gate * c_prev + in_gate * c_new
    h_curr = out_gate * cgt.tanh(c_curr)
    return c_curr, h_curr

# def vanilla_block(h_prev, c_prev, x_curr, size_x, size_c, name=''):
#     c_curr = nn.Affine(size_x, size_c, name=name+'*x')(x_curr) + \
#                  nn.Affine(size_c, size_c, name=name+'*h')(c_prev)
#     c_curr = cgt.tanh(input_sums)
#     h_curr = nn.Affine(size_c, size_c, name=name+'*y')(c_curr)
#     return c_curr, h_curr
