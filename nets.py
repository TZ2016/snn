import cgt
from cgt import nn
from layers import lstm_block, combo_layer


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

# def lstm_layers(X, size_in, size_out, num_units):
#     """
#     Construct a recurrent neural network with multiple layers of LSTM units,
#     with each layer a block of cells sharing a common set of gate units.
#     Return list of inputs and a list of outputs for the net at one time step.
#     Inputs =  [ net_in, hidden layers ]
#     Outputs = [ hidden layers, net_out ]
#
#     :param size_in: input dimension
#     :param num_units: number of memory units for each layer
#     :param size_out: output dimension
#     :return:
#     :rtype: (int, list, list)
#     """
#     net_in = X
#     net_c_prev, net_h_prev = [], []
#     net_c_curr, net_h_curr = [], []
#     prev_l_num_units, prev_out = size_in, net_in
#     for l_num_units in num_units:
#         c_prev = cgt.matrix(fixed_shape=(None, l_num_units))
#         h_prev = cgt.matrix(fixed_shape=(None, l_num_units))
#         c_curr, h_curr = lstm_block(h_prev, c_prev, prev_out,
#                                     prev_l_num_units, l_num_units)
#         net_c_prev.append(c_prev)
#         net_h_prev.append(h_prev)
#         net_c_curr.append(c_curr)
#         net_h_curr.append(h_curr)
#         prev_l_num_units = l_num_units
#         prev_out = h_curr
#     inputs = [net_in] + net_c_prev + net_h_prev
#     outputs = net_c_curr + net_h_curr
#     return inputs, outputs
#     # return (net_in, net_c_prev, net_h_prev), (net_c_curr, net_h_curr)
