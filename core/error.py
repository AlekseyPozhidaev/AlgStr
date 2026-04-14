import numpy as np

from core.spline import spline

def get_spline_error(func, x_nodes, dif=500):
    a, b = min(x_nodes), max(x_nodes)
    x = np.linspace(a, b, dif)
    y = func(x)
    y_nodes = func(x_nodes)

    spl = spline(x_nodes, y_nodes, x)
    err_spline = np.max(np.abs(y - spl))

    return err_spline