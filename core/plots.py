import matplotlib.pyplot as plt
import numpy as np

from core.spline import spline
from core.error import get_spline_error

def spline_plot(n: int, func, dots=500) -> None:
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = func(x_nodes)

    x_dots = np.linspace(-1, 1, dots)
    y_dots = func(x_dots)

    y_spline = spline(x_nodes=x_nodes, y_nodes=y_nodes, x_dots=x_dots)

    plt.plot(x_dots, y_dots, label="func")
    plt.plot(x_dots, y_spline, label="spline")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(fontsize=11)
    plt.grid(True)

    plt.show()

def error_plot(func, dots=500):
    errors = []

    for n in range(5, 15):
        x_nodes = np.linspace(-1, 1, n)
        errors.append(get_spline_error(func, x_nodes, dots))

    plt.plot(range(5, 15), errors, label="error")

    plt.xlabel('n')
    plt.ylabel('max error')
    plt.legend(fontsize=11)
    plt.grid(True)

    plt.show()