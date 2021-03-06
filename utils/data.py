__all__ = [
    'scale_data',
    'data_synthetic_a',
    'data_sigm_multi',
    'data_add',
    'data_seq',
]

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.special import expit as sigmoid


def scale_data(X, Y, Y_var=None, scalers=None):
    if not scalers:
        scalers = [StandardScaler() for _ in range(2)]
    assert len(scalers) == 2
    s_X = scalers[0].fit_transform(X)
    s_Y = scalers[1].fit_transform(Y)
    if Y_var is not None:
        scalers[1].with_mean = False
        s_Y_var = np.square(scalers[1].transform(np.sqrt(Y_var)))
        scalers[1].with_mean = True
        return s_X, s_Y, s_Y_var
    return s_X, s_Y


def generate_examples(N, x, y, p_y):
    X = x * np.ones((N, x.size))
    Y = y * np.ones((N, y.size))
    for i, p in enumerate(p_y):
        if p is not None:
            Y[:, i] = 0.
            Y[:, i][:int(N*p)] = 1.
    np.random.shuffle(Y)
    return X, Y


def data_synthetic_a(N):
    # x = y + 0.3 sin(2 * pi * y) + e, e ~ Unif(-0.1, 0.1)
    Y = np.random.uniform(0., 1., N)
    X = Y + 0.3 * np.sin(2. * Y * np.pi) + np.random.uniform(-.1, .1, N)
    Z = np.random.normal(scale=.05, size=N)
    # X = np.hstack([X, Z])
    # Y = np.hstack([Y, Z])
    i_s =np.argsort(X, axis=0)
    Y, X = np.reshape(Y[i_s], (N, 1)), np.reshape(X[i_s], (N, 1))
    Y_var = np.array([np.var(Y[i:i+20]) for i in range(N-20)] +
                     [np.var(Y[N-20:])] * 20).reshape((N, 1))
    return X, Y, Y_var


def data_sigm_multi(N, p):
    if not isinstance(p, (list, tuple)):
        assert isinstance(p, int)
        p = [1. / p] * p
    X = np.random.uniform(-10., 10., N)
    Y = sigmoid(X)
    Y += np.random.normal(0., .1, N)
    y = np.random.multinomial(1, p, size=N)
    Y += np.nonzero(y)[1]
    Y, X = Y.reshape((N, 1)), X.reshape((N, 1))
    return X, Y


def data_add(N, T, k=2, dim=1):
    Xs, Ys = [], []
    for _ in range(N):
        # X = np.random.normal(0., 1., T)
        X = np.random.binomial(1, 0.5, size=(T, dim))
        if k == 1:
            Y = X.copy()
        elif k == 2:
            Y = np.concatenate((X[[0]], X[1:] + X[:-1]))
        else:
            Y = None
        Y, X = Y.reshape((T, dim)), X.reshape((T, dim))
        Xs.append(X)
        Ys.append(Y)
    Xs, Ys = np.array(Xs, dtype=float), np.array(Ys, dtype=float)
    return Xs, Ys


def data_seq(N, T):
    lag = 5
    Xs, Ys = [], []
    for _ in range(N):
        start = float(np.random.uniform(0., 2. * np.pi))
        t = np.linspace(start, start + 2 * np.pi, num=T+lag)
        X = np.sin(t[:-lag])
        Y = np.r_[np.sin(t[lag:])]
        # Y += np.random.normal(0., 0.1, T)
        # X += np.random.normal(0., 0.1, T)
        Y, X = Y.reshape((T, 1)), X.reshape((T, 1))
        Xs.append(X)
        Ys.append(Y)
    Xs, Ys = np.array(Xs, dtype=float), np.array(Ys, dtype=float)
    return Xs, Ys
