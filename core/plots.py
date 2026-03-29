import matplotlib.pyplot as plt
import numpy as np

from core.chebyshev import chebyshev_nodes
from core.lagrange import lagrange
from core.newton import newton
from core.spline import spline
from core.error import get_lagrange_error, get_newton_error, get_spline_error


def newton_lagrange_spline_uniform(func, x_nodes=None, dif=100):
    if x_nodes is None: x_nodes=[-2, -1, 0, 1, 2]
    y_nodes = func(x_nodes)

    x = np.linspace(min(x_nodes), max(x_nodes), dif)
    y = func(x)

    lagrangian_uniform = lagrange(x_nodes=x_nodes, y_nodes=y_nodes, x=x)
    uniform_lagrange_error = (y - lagrangian_uniform)

    newton_uniform = newton(x_nodes=x_nodes, y_nodes=y_nodes, x=x)
    uniform_newton_error = (y - newton_uniform)

    spline_uniform = spline(x_nodes=x_nodes, y_nodes=y_nodes, x_dots=x)
    uniform_spline_error = (y - spline_uniform)

    fig, axs = plt.subplots(3, 2)

    #Spline
    axs[0, 0].plot(x, y, label="func")
    axs[0, 0].plot(x, spline_uniform, label="spline")
    axs[0, 0].set_title('Spline')
    axs[0, 1].plot(x, uniform_spline_error)
    axs[0, 1].set_title('Spline uniform error')
    #Newton
    axs[1, 0].plot(x, y, label="func")
    axs[1, 0].plot(x, newton_uniform, label="Newton")
    axs[1, 0].set_title('Newton')
    axs[1, 1].plot(x, uniform_newton_error)
    axs[1, 1].set_title('Newton uniform error')
    #Lagrange
    axs[2, 0].plot(x, y, label="func")
    axs[2, 0].plot(x, lagrangian_uniform, label="Lagrange")
    axs[2, 0].set_title('Lagrange')
    axs[2, 1].plot(x, uniform_lagrange_error)
    axs[2, 1].set_title('Lagrange uniform error')

    for ax in axs.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=11)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def newton_lagrange_spline_chebyshev(func, ch_deg=5, left=-1, right=1, dif=100):
    x_nodes = chebyshev_nodes(ch_deg, left,right)
    y_nodes = func(x_nodes)

    x = np.linspace(min(x_nodes), max(x_nodes), dif)
    y = func(x)

    lag = lagrange(x_nodes=x_nodes, y_nodes=y_nodes, x=x)
    lagrange_error = (y - np.asarray(lag))

    newt = newton(x_nodes=x_nodes, y_nodes=y_nodes, x=x)
    newton_error = (y - np.asarray(newt))

    spl = spline(x_nodes=x_nodes, y_nodes=y_nodes, x_dots=x)
    uniform_spline_error = (y - np.asarray(spl))

    fig, axs = plt.subplots(3, 2)

    # Spline
    axs[0, 0].plot(x, y, label="func")
    axs[0, 0].plot(x, spl, label="spline")
    axs[0, 0].set_title('Spline')
    axs[0, 1].plot(x, uniform_spline_error)
    axs[0, 1].set_title('Spline Chebyshev error')
    # Newton
    axs[1, 0].plot(x, y, label="func")
    axs[1, 0].plot(x, newt, label="Newton")
    axs[1, 0].set_title('Newton')
    axs[1, 1].plot(x, newton_error)
    axs[1, 1].set_title('Newton Chebyshev error')
    # Lagrange
    axs[2, 0].plot(x, y, label="func")
    axs[2, 0].plot(x, lag, label="Lagrange")
    axs[2, 0].set_title('Lagrange')
    axs[2, 1].plot(x, lagrange_error)
    axs[2, 1].set_title('Lagrange Chebyshev error')

    for ax in axs.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=11)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def newton_lagrange_spline_error(func, n_min=3, n_max=15, dif=100):
    lag_un_err, lag_ch_err, newt_un_err, newt_ch_err, spl_un_err, spl_ch_err = [],[],[],[],[],[]
    n_range = np.linspace(n_min, n_max, n_max-n_min+1)

    for n in range(n_min, n_max + 1):
        x_nodes = np.linspace(-1, 1, n)
        lue, lce = get_lagrange_error(func,x_nodes,dif)
        nue, nce = get_newton_error(func,x_nodes,dif)
        sue, sce = get_spline_error(func,x_nodes,dif)
        lag_un_err.append(lue)
        lag_ch_err.append(lce)
        newt_un_err.append(nue)
        newt_ch_err.append(nce)
        spl_un_err.append(sue)
        spl_ch_err.append(sce)

    fig, axs = plt.subplots(3)

    axs[0].plot(n_range, spl_un_err, label="Uniform")
    axs[0].plot(n_range, spl_ch_err, label="Chebyshev")
    axs[0].set_title('Spline error')

    axs[1].plot(n_range, newt_un_err, label="Uniform")
    axs[1].plot(n_range, newt_un_err, label="Chebyshev")
    axs[1].set_title('Newton error')

    axs[2].plot(n_range, lag_un_err, label="Uniform")
    axs[2].plot(n_range, lag_un_err, label="Chebyshev")
    axs[2].set_title('Lagrange error')

    for ax in axs.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=11)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def uniform_chebyshev_spline(func, x_nodes=None, dif=100):
    if x_nodes is None:
        x_nodes = np.array([-2, -1, 0, 1, 2])
    y_nodes = func(x_nodes)

    # Чебышёвские узлы
    x_ch = chebyshev_nodes(len(x_nodes), min(x_nodes), max(x_nodes))
    y_ch = func(x_ch)

    # Общая сетка для построения (от min(x_nodes) до max(x_nodes))
    x = np.linspace(min(x_nodes), max(x_nodes), dif)
    y = func(x)

    # Равномерный сплайн (работает на всём интервале)
    spline_uniform = spline(x_nodes, y_nodes, x)
    uniform_error = y - spline_uniform

    # Чебышёвский сплайн – только внутри его узлов
    spline_chebyshev = np.full_like(x, np.nan)
    mask = (x >= min(x_ch)) & (x <= max(x_ch))
    if np.any(mask):
        spline_chebyshev[mask] = spline(x_ch, y_ch, x[mask])
    chebyshev_error = y - spline_chebyshev

    fig, axs = plt.subplots(2)
    axs[0].plot(x, y, label="func")
    axs[0].plot(x, spline_uniform, label="Uniform")
    axs[0].plot(x, spline_chebyshev, label="Chebyshev")
    axs[0].set_title('Interpolation')
    axs[0].legend()

    axs[1].plot(x, uniform_error, label="Uniform")
    axs[1].plot(x, chebyshev_error, label="Chebyshev")
    axs[1].set_title('Error')
    axs[1].legend()

    for ax in axs:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)

    plt.tight_layout()
    plt.show()