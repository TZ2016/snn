__all__ = ['safe_io', 'example_debug']


import os
import warnings
import matplotlib.pyplot as plt
import numpy as np


def safe_io(func, base_path, rel_path=None, flag=None):
    abs_path = base_path if rel_path is None else os.path.join(base_path, rel_path)
    d = os.path.dirname(abs_path)
    if not os.path.exists(d):
        warnings.warn("Making new directory: %s" % d)
        os.makedirs(d)
    if flag == 'r':
        if not os.path.isfile(abs_path):
            warnings.warn("Attempt to read non-existing file: %s" % abs_path)
            flag = 'w'
    elif flag == 'w':
        if os.path.exists(abs_path):
            warnings.warn("Overwritten an existing file: %s" % abs_path)
    with open(abs_path, flag) as f:
        func(f)


# def safe_path(base_path, rel_path=None, flag=None):
#     abs_path = base_path if rel_path is None else os.path.join(base_path, rel_path)
#     d = os.path.dirname(abs_path)
#     if not os.path.exists(d):
#         warnings.warn("Making new directory: %s" % d)
#         os.makedirs(d)
#     if flag == 'r':
#         if os.path.isfile(abs_path):
#             return open(abs_path, 'r')
#         warnings.warn("Attempt to read non-existing file: %s" % abs_path)
#         return open(abs_path, 'w')
#     elif flag == 'w':
#         if os.path.exists(abs_path):
#             warnings.warn("Overwritten an existing file: %s" % abs_path)
#         return open(abs_path, 'w')
#     return abs_path


def example_debug(args, X, Y, Y_var=None):
    # TODO shape mismatch
    def h_ax(ax, title=None, x=None, y=None):
        if not isinstance(ax, (tuple, list, np.ndarray)): ax = [ax]
        for a in ax:
            if title: a.set_title(title)
            if x:
                if x == 'hide':
                    plt.setp(a.get_xticklabels(), visible=False)
                elif x == 'mm':  # min/max
                    plt.setp(a.get_xticklabels()[1:-1], visible=False)
                    plt.setp(a.get_xticklabels()[0], visible=True)
                    plt.setp(a.get_xticklabels()[-1], visible=True)
                elif x == 'auto':
                    if len(a.get_xticklabels()) > 20:
                        plt.setp(a.get_xticklabels(), visible=False)
                        plt.setp(a.get_xticklabels()[::5], visible=True)
                else:
                    raise KeyError
            if y:
                if y == 'hide':
                    plt.setp(a.get_yticklabels(), visible=False)
                elif y == 'mm':  # min/max
                    plt.setp(a.get_yticklabels()[1:-1], visible=False)
                    plt.setp(a.get_yticklabels()[0], visible=True)
                    plt.setp(a.get_yticklabels()[-1], visible=True)
                else:
                    raise KeyError
    X, Y, Y_var = np.squeeze(X, axis=1), np.squeeze(Y, axis=1), np.squeeze(Y_var, axis=1)
    N, _ = X.shape
    _safe_io = lambda func, p: safe_io(func, p, args['dump_path'])
    conv_smoother = lambda x: np.convolve(x, [1. / N] * N, mode='valid')
    _ix, _iy = args['dbg_plot_samples']['x_dim'], args['dbg_plot_samples']['y_dim']
    # cache
    ep_net_distr = []
    it_loss_surr = []
    it_grad_norm, it_grad_norm_comp = [], []
    it_theta_norm, it_theta_norm_comp = [], []
    def dbg_iter(num_epochs, num_iters, info, workspace):
        optim_state = workspace['optim_state']
        param_col = workspace['param_col']
        f_step = workspace['f_step']
        it_loss_surr.append(info['objective'])
        it_grad_norm.append(np.linalg.norm(optim_state['scratch']))
        it_grad_norm_comp.append([np.linalg.norm(g) / np.size(g)
                                  for g in info['grad']])
        it_theta_norm.append(np.linalg.norm(optim_state['theta']))
        it_theta_norm_comp.append([np.linalg.norm(t) / np.size(t)
                                   for t in param_col.get_values()])
        if num_iters == 0:  # new epoch
            print "Epoch %d" % num_epochs
            print "Mean gradient norm = %f" % np.mean(it_grad_norm[-N:])
            print "Mean theta norm = %f" % np.mean(it_theta_norm[-N:])
            print "Mean objective = %f" % np.mean(it_loss_surr[-N:])
            if args['dbg_plot_samples']['plot']:
                s_ind = np.random.choice(N, size=args['dbg_plot_samples']['batch'], replace=False)
                s_X, s_Y_var = X[s_ind, :], 1. / Y_var[s_ind, :]
                s_Y_mean = f_step(s_X)[0]
                err_plt = lambda: plt.errorbar(s_X[:, _ix], s_Y_mean[:, _iy],
                                               yerr=np.sqrt(s_Y_var[:, _iy]),
                                               fmt='none')
                ep_net_distr.append((num_epochs, err_plt))
    def dbg_done(workspace):
        kw_ticks = {
            'xticks': np.arange(args['n_epochs']) * N,
            'xticklabels': np.arange(args['n_epochs']).astype(str)
        }
        # plot overview
        f, axs = plt.subplots(3, 1, sharex=True, subplot_kw=kw_ticks)
        f.suptitle('Training Overview')
        axs[0].plot(conv_smoother(it_loss_surr))
        h_ax(axs[0], title='Objective', x='hide')
        axs[1].plot(conv_smoother(it_grad_norm)); axs[1].set_title('grad')
        h_ax(axs[1], title='Gradient Norm', x='hide')
        axs[2].plot(conv_smoother(it_theta_norm)); axs[2].set_title('theta')
        h_ax(axs[2], title='Theta Norm', x='auto')
        _safe_io(lambda file: f.savefig(file), 'overview.png'); plt.close(f)
        if False and args['dbg_plot_charts']:
            # plot grad norm component-wise
            _grad_norm_cmp = np.array(it_grad_norm_comp).T
            f, axs = plt.subplots(_grad_norm_cmp.shape[0], 1, sharex=True, subplot_kw=kw_ticks)
            f.suptitle('grad norm layer-wise')
            for _i, _ax in enumerate(axs):
                _ax.plot(conv_smoother(_grad_norm_cmp[_i]))
            h_ax(axs[:-1], x='hide'); h_ax(axs, y='mm')
            f.tight_layout(); _safe_io(lambda fi: f.savefig(fi), 'norm_grad_cmp.png'); plt.close(f)
            # plot theta norm component-wise
            _theta_norm_cmp = np.array(it_theta_norm_comp).T
            f, axs = plt.subplots(_theta_norm_cmp.shape[0], 1, sharex=True, subplot_kw=kw_ticks)
            f.suptitle('theta norm layer-wise')
            for _i, _ax in enumerate(axs):
                _ax.plot(conv_smoother(_theta_norm_cmp[_i]))
            h_ax(axs[:-1], x='hide'); h_ax(axs, y='mm')
            f.tight_layout(); _safe_io(lambda fi: f.savefig(fi), 'norm_theta_cmp.png'); plt.close(f)
        # plot samples for each epoch
        if args['dbg_plot_samples']['plot']:
            for _e, _distr in enumerate(ep_net_distr):
                _ttl = 'epoch_%d.png' % _e
                plt.scatter(X[:, _ix], Y[:, _iy], alpha=0.5, color='y', marker='*')
                plt.gca().set_autoscale_on(False)
                _distr[1](); plt.title(_ttl)
                _safe_io(lambda f: plt.savefig(f), '_sample/' + _ttl); plt.cla()
    return dbg_iter, dbg_done
