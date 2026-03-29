import numpy as np

from core.chebyshev import chebyshev_nodes
from core.spline import spline
from core.lagrange import lagrange
from core.newton import newton

def get_lagrange_error(func, x_nodes, dif=500):
    a, b = min(x_nodes), max(x_nodes)
    x = np.linspace(a, b, dif)
    y = func(x)

    # Лагранж равномерный
    y_nodes = func(x_nodes)
    lag_uniform = lagrange(x_nodes, y_nodes, x)
    err_lag_uniform = np.max(np.abs(y - lag_uniform))

    # Лагранж Чебышев
    x_cheb = chebyshev_nodes(len(x_nodes), a, b)
    y_cheb = func(x_cheb)
    lag_cheb = lagrange(x_cheb, y_cheb, x)
    err_lag_cheb = np.max(np.abs(y - lag_cheb))

    return err_lag_uniform, err_lag_cheb


def get_newton_error(func, x_nodes, dif=500):
    a, b = min(x_nodes), max(x_nodes)
    x = np.linspace(a, b, dif)
    y = func(x)
    y_nodes = func(x_nodes)

    newt_uniform = newton(x_nodes, y_nodes, x)
    err_newton = np.max(np.abs(y - newt_uniform))

    x_cheb = chebyshev_nodes(len(x_nodes), a, b)
    y_cheb = func(x_cheb)
    x_cheb_lin = np.linspace(min(x_cheb), max(x_cheb), dif)
    y_cheb_true = func(x_cheb_lin)
    newt_cheb = newton(x_cheb, y_cheb, x_cheb_lin)
    err_newt_cheb = np.max(np.abs(y_cheb_true - newt_cheb))

    return err_newton, err_newt_cheb

def get_spline_error(func, x_nodes, dif=500):
    a, b = min(x_nodes), max(x_nodes)
    x = np.linspace(a, b, dif)
    y = func(x)
    y_nodes = func(x_nodes)

    spl = spline(x_nodes, y_nodes, x)
    err_spline = np.max(np.abs(y - spl))

    # Чебышёвские узлы и своя сетка для них
    x_cheb = chebyshev_nodes(len(x_nodes), a, b)
    y_cheb = func(x_cheb)
    x_cheb_lin = np.linspace(min(x_cheb), max(x_cheb), dif)   # ← новая сетка
    y_cheb_true = func(x_cheb_lin)                             # ← истинные значения на этой сетке
    spline_cheb = spline(x_cheb, y_cheb, x_cheb_lin)
    err_spline_cheb = np.max(np.abs(y_cheb_true - spline_cheb))

    return err_spline, err_spline_cheb