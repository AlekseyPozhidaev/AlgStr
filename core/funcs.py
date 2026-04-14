import numpy as np

import numpy as np

def f1(x):
    """y = sin(5x) — осциллирующая функция"""
    x = np.array(x)
    return np.sin(5 * x)

def f2(x):
    """y = e^{-x^2} — гладкая колоколообразная функция"""
    x = np.array(x)
    return np.exp(-x ** 2)

def f3(x):
    """y = arctan(10x) — функция с резким переходом (ступенька)"""
    x = np.array(x)
    return np.arctan(10 * x)