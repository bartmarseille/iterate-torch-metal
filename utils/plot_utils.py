import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import numbers

from utils.str_utils import to_str

PLOT_FIGSIZE_DEFAULT = figsize=(4, 4)

def init_darkmode():
    mpl.style.use("dark_background")
    # prepend yellow to color_cycle
    color_cycle=['y']+[d['color'] for d in mpl.rcParams['axes.prop_cycle']]
    # mpl.rcParams['axes.prop_cycle'] = cycler(color=color_cycle[:-1])
    mpl.rcParams['axes.prop_cycle'] = cycler(color=color_cycle)


def plot_map(map, P, lim=(0,1), ax=None):

    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))
    (x_min, x_max) = lim

    # plot x and y axis
    if x_min < 0 < x_max:
        ax.plot([x_min, x_max], [0, 0], c='w', lw=0.5)
        ax.plot([0, 0], [x_min, x_max], c='w', lw=0.5)

    # plot mapping
    x = np.linspace(x_min, x_max)
    y = map(x, P)
    ax.plot(x, y, c='b', lw=2)
    ax.plot(x, x, c='b', lw=1)

    ax.set_xlabel('$x_{t}$')
    ax.set_ylabel('$x_{t+1}$')
    ax.set_title(f'$P$={to_str(P)}')
    ax.set_xlim(lim)
    ax.set_ylim(lim)

def plot_cobweb(map, T, P, ax=None):

    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

    # plot mapping
    x = np.linspace(0, 1)
    y = map(x, P)
    ax.plot(x, y, c='b', lw=2)
    ax.plot(x, x, c='b', lw=1)

    # plot iteration
    n = len(T) if T is not None else 0
    if n is None:
        ax.set_title(f'$P$={to_str(P):.1f}')
    else:
        if T.size==0 or n==0:
            ax.plot([T[0]], [T[0]], 'oy', ms=5)
            ax.set_title(f'$P$={to_str(P, 1):.1f}, $x_0$={T[0]:.2g}')
        else:
            # Recursively apply y=f(x) and plot two lines:
            # (x, x) -> (x, y)
            # (x, y) -> (y, y)
            for i in range(n-1):
                ax.plot([T[i], T[i]], [T[i], T[i+1]], c='y', lw=1)
                ax.plot([T[i]], [T[i+1]], 'oy', ms=5, alpha=(i+1)/n)
                ax.plot([T[i], T[i+1]], [T[i+1], T[i+1]], c='y', lw=1)
            ax.plot([T[-2], T[-2]], [T[-2], T[-1]], c='y', lw=1)
            ax.plot([T[0]], [T[0]], 'or', ms=5)
            c = 'g' if n>1 else 'y'
            ax.plot([T[-2]], [T[-1]], color=c, marker='o', ms=5)
            ax.set_title(f'$P$={to_str(P, 6)}, $x_0$={T[0]:.2g}, $n$={n}')
            ax.set_ylabel('$x_{t+1}$')
    ax.set_xlabel('$x_{t}$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_trajectory(T, P, ax=None, figsize=(16,4), labels=False):
    
    if ax:
        result = ax
    else:
        result = None
        _, ax = plt.subplots(1, 1, figsize=figsize)
    
    n = len(T) if T is not None else 0
    ax.plot(T, marker='.', lw=1)
    if labels:
        n_decimal_map = [[55, None], [36, 1], [28,2], [14,3]]
        n_dec = next(iter(i[1] for i in n_decimal_map if i[0]<=n), 4)
        if n_dec:
            for i, t in enumerate(T):
                # Annotate the points 5 _points_ above and to the left of the vertex
                ax.annotate('{0:.{1}g}'.format(t, n_dec), xy=(i,t), xytext=(-5, 5), ha='right', textcoords='offset points')
    ax.set_xlabel(f'$time$ $→$')
    ax.set_ylabel(f'$x_t$')
    title = f'$P$={to_str(P)}, $x_0$={T[0]:.2g}, $n$={n}'
    ax.set_title(title)
    if n < 25:
        ax.set_xticks(range(0, n+1))
    return result


def plot_trajectory_and_cobweb(map, T, P, axes=None, labels=False):
    if axes is not None:
        (ax1, ax2) = axes
        result = axes
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
        result = None
    plot_trajectory(T, P, ax=ax1, labels=labels)
    plot_cobweb(map, T, P, ax=ax2)
    if result is None:
        plt.tight_layout()
    return result

def plot_trajectories(Ts, Ps, ax=None, figsize=(16,4), labels=False):

    if ax:
        result = ax
    else:
        result = None
        _, ax = plt.subplots(1, 1, figsize=figsize)
    
    if not isinstance(Ps, (list, np.ndarray)):
        n_T = len(Ts) if Ts is not None else 0
        Ps = np.repeat(Ps, n_T)

    for i, T in enumerate(Ts):
        n = len(T) if T is not None else 0
        ax.plot(T, marker='.', lw=1,label=f'$P$={to_str(Ps[i])}, $x_0$={T[0]:.1f}, $n$={n}')
    ax.set_xlabel(f'$time$ $→$')
    ax.set_ylabel(f'$x_t$')

    ax.legend(loc="upper left")
    if result is None:
        plt.tight_layout()
    return result

def plot_distribution(T, P, bins=100, range=(0, 1), t_label='', n_dec=4, ax=None):
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(16, 4))

    n = len(T) if T is not None else 0
    (dist, bins, patches) = ax.hist(x=T, bins=bins, range=range, edgecolor = 'gray', align='mid')
    ax.set_xlabel('$x_{t}$')
    ax.set_ylabel('$x_{t+1}$')
    ax.set_title(f'$T_{{{t_label}}}$ distribution: $P$={to_str(P, n_dec)}, $x_0$={T[0]:.2g}, $n$={n}')
    ax.set_xlim((0, 1))
    return ax

# def plot_distributions(Ts, Ps, bins=100, range=(0, 1), ax=None, figsize=(16,4), labels=False):

#     n = len(Ts) if Ts is not None else 0

#     if ax:
#         result = ax
#     else:
#         result = None
#         _, axis = plt.subplots(1, n, figsize=figsize)
    
    

#     if not isinstance(Ps, (list, np.ndarray)):
#         Ps = np.repeat(Ps, n)

#     for i, T in enumerate(Ts):
#         n = len(T) if T is not None else 0
#         ax.plot(T, marker='.', lw=1,label=f'$P$={to_str(P[i])}, $x_0$={T[0]:.1f}, $n$={n}')
#     ax.set_xlabel(f'$time$ $→$')
#     ax.set_ylabel(f'$x_t$')

#     # Setting the values for all axes.
#     plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)

#     plt.legend(loc="upper left")
#     if result is None:
#         plt.tight_layout()
#     return result


def plot_loss(L, loss_type='Actual', ax=None):
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(16, 4))

    ax.plot(L)
    ax.set_title(f'{loss_type.capitalize()} Loss - final loss: ${L[-1]: .4f}$')
    ax.set_xlabel('$epochs$')
    ax.set_ylabel('$loss$')
    # ax.savefig('deep_ae_mnist_loss.png', facecolor='white', transparent=False)
    return ax
