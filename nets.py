import cgt
from layers import lstm_block


# TODO untested
def lstm_layers(X, size_in, size_out, num_units):
    """
    Construct a recurrent neural network with multiple layers of LSTM units,
    with each layer a block of cells sharing a common set of gate units.
    Return list of inputs and a list of outputs for the net at one time step.
    Inputs =  [ net_in, hidden layers ]
    Outputs = [ hidden layers, net_out ]

    :param size_in: input dimension
    :param num_units: number of memory units for each layer
    :param size_out: output dimension
    :return:
    :rtype: (int, list, list)
    """
    net_in = X
    net_c_prev, net_h_prev = [], []
    net_c_curr, net_h_curr = [], []
    prev_l_num_units, prev_out = size_in, net_in
    for l_num_units in num_units:
        c_prev = cgt.matrix(fixed_shape=(None, l_num_units))
        h_prev = cgt.matrix(fixed_shape=(None, l_num_units))
        c_curr, h_curr = lstm_block(h_prev, c_prev, prev_out,
                                    prev_l_num_units, l_num_units)
        net_c_prev.append(c_prev)
        net_h_prev.append(h_prev)
        net_c_curr.append(c_curr)
        net_h_curr.append(h_curr)
        prev_l_num_units = l_num_units
        prev_out = h_curr
    inputs = [net_in] + net_c_prev + net_h_prev
    outputs = net_c_curr + net_h_curr
    return inputs, outputs
    # return (net_in, net_c_prev, net_h_prev), (net_c_curr, net_h_curr)

