import numpy as np


def safe_logadd(lst):
    m = max(lst)
    s = np.log(np.sum(np.exp(np.array(lst) - m))) + m
    return s