import traceback

import numpy as np


def spline(x_nodes, y_nodes, x_dots):
    try:
        if min(x_dots) < x_nodes[0] or max(x_dots) > x_nodes[-1]:
            raise ValueError

        if not np.all(np.diff(x_nodes) > 0):
            raise ValueError("x_nodes must be strictly increasing")

        h = get_dist(x_nodes)
        m = [0] + second_dir(h, y_nodes) + [0]
        y = []

        for x in x_dots:
            i = 0
            while i < len(x_nodes) - 1 and x > x_nodes[i + 1]:
                i += 1

            y.append((m[i]*(x_nodes[i+1]-x)**3)/(6*h[i]) +
                 (m[i+1]*(x-x_nodes[i])**3)/(6*h[i]) +
                 ((y_nodes[i]/h[i]) - (m[i]*h[i])/6)*(x_nodes[i+1]-x) +
                 ((y_nodes[i+1]/h[i]) - (m[i+1]*h[i])/6)*(x-x_nodes[i]))

        return np.asarray(y)

    except ValueError as e:
        print("=== ОШИБКА В СПЛАЙНЕ ===")
        print("Сообщение:", str(e))
        traceback.print_exc()  #
        return None

def get_dist(x_nodes):
    h = []
    for i in range(len(x_nodes)-1):
        h.append(x_nodes[i+1]-x_nodes[i])
    return h

def second_dir(h, y_nodes):
    a, b, c, d = tridiagonal(h, y_nodes)
    m = solve_tridiagonal(a, b, c ,d)
    return m

def tridiagonal(h, y_nodes):
    a, b, c, d = [], [], [], []
    for i in range(1, len(h)):
        a.append(h[i-1]/6)
        b.append((h[i-1]+h[i])/3)
        c.append(h[i]/6)
        d.append((y_nodes[i+1]-y_nodes[i])/h[i]-(y_nodes[i]-y_nodes[i-1])/h[i-1])
    return a, b, c, d


def solve_tridiagonal(a, b, c, d):
    n = len(d)

    alpha = [0] * n
    beta = [0] * n

    # прямой ход
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denom = a[i] * alpha[i - 1] + b[i]
        alpha[i] = -c[i] / denom
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denom

    # обратный ход
    x = [0] * n
    x[-1] = beta[-1]

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x