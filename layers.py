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
        if _f_s is None:
            _i = cgt.sigmoid(nn.Affine(size_in, _s_out, name=_name)(X))
        else:
            _i = _f_s(X, size_in, _s_out, name=_name)
        _o = _f_o(_i) if _f_o is not None else _i
        curr = _split
        ins.append(_i)
        outs.append(_o)
        names.append(_name)
    out = cgt.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]
    dbg_out.update(dict(zip([_n + '~in' for _n in names], ins)))
    dbg_out.update(dict(zip([_n + '~out' for _n in names], outs)))
    return out
