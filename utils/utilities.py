import numpy as np


def safe_logadd(lst):
    m = max(lst)
    s = np.log(np.sum(np.exp(np.array(lst) - m))) + m
    return s


class _Placeholder(object):
    def __getitem__(self, item):
        return self

    def __setitem__(self, key, value):
        raise RuntimeError

    def __getattribute__(self, item):
        return self

    def __call__(self, *args, **kwargs):
        return self


NONE = _Placeholder()
