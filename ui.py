import sys
import numpy as np
from functools import partial

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QGroupBox,
    QFormLayout, QMessageBox
)
from PyQt6.QtCore import Qt

# ====================== Твои модули ======================
from core.funcs import (
    linear, quadratic, cubic, sine_wave, damped_cosine,
    exponential, logarithm, gaussian, sigmoid
)
from core.plots import (
    newton_lagrange_spline_uniform,
    newton_lagrange_spline_chebyshev,
    newton_lagrange_spline_error,
    uniform_chebyshev_spline
)


class InterpolationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Интерполяция: Лагранж • Ньютон • Сплайн")
        self.setGeometry(100, 100, 640, 820)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(15)

        # ====================== Выбор функции ======================
        func_group = QGroupBox("Выберите функцию")
        func_layout = QVBoxLayout()
        func_group.setLayout(func_layout)

        self.func_combo = QComboBox()
        self.available_funcs = {
            'linear': linear,
            'quadratic': quadratic,
            'cubic': cubic,
            'sine_wave': sine_wave,
            'damped_cosine': damped_cosine,
            'exponential': exponential,
            'logarithm': logarithm,
            'gaussian': gaussian,
            'sigmoid': sigmoid
        }
        self.func_combo.addItems(self.available_funcs.keys())
        self.func_combo.currentTextChanged.connect(self.update_params)
        func_layout.addWidget(self.func_combo)

        # Параметры функции
        self.param_group = QGroupBox("Параметры функции")
        self.param_layout = QFormLayout()
        self.param_group.setLayout(self.param_layout)
        self.param_entries = {}

        # ====================== Общие параметры ======================
        common_group = QGroupBox("Общие параметры")
        common_layout = QFormLayout()

        self.a_entry = QLineEdit("-1")
        self.b_entry = QLineEdit("1")
        self.n_entry = QLineEdit("10")
        self.ch_entry = QLineEdit("5")          # используется как ch_deg
        self.nmin_entry = QLineEdit("3")
        self.nmax_entry = QLineEdit("15")

        common_layout.addRow("Интервал a:", self.a_entry)
        common_layout.addRow("Интервал b:", self.b_entry)
        common_layout.addRow("Число узлов n:", self.n_entry)
        common_layout.addRow("Степень Чебышева (ch_deg):", self.ch_entry)
        common_layout.addRow("Ошибка vs n (от — до):", self.nmin_entry)
        h = QHBoxLayout()
        h.addWidget(self.nmin_entry)
        h.addWidget(QLabel("—"))
        h.addWidget(self.nmax_entry)
        common_layout.addRow("", h)

        common_group.setLayout(common_layout)

        # ====================== Кнопки ======================
        btn_layout = QVBoxLayout()
        btn_style = "QPushButton { padding: 12px; font-size: 14px; font-weight: bold; }"

        self.btn_uniform = QPushButton("1. Uniform (Лагранж + Ньютон + Сплайн)")
        self.btn_chebyshev = QPushButton("2. Chebyshev (Лагранж + Ньютон + Сплайн)")
        self.btn_error = QPushButton("3. Зависимость ошибки от n")
        self.btn_spline_compare = QPushButton("4. Сравнение сплайна Uniform vs Chebyshev")

        for btn in (self.btn_uniform, self.btn_chebyshev, self.btn_error, self.btn_spline_compare):
            btn.setStyleSheet(btn_style)
            btn_layout.addWidget(btn)

        self.btn_uniform.clicked.connect(self.run_uniform)
        self.btn_chebyshev.clicked.connect(self.run_chebyshev)
        self.btn_error.clicked.connect(self.run_error)
        self.btn_spline_compare.clicked.connect(self.run_spline_compare)

        # ====================== Сборка интерфейса ======================
        main_layout.addWidget(func_group)
        main_layout.addWidget(self.param_group)
        main_layout.addWidget(common_group)
        main_layout.addLayout(btn_layout)
        main_layout.addStretch()

        self.update_params()

    # ====================== Обновление параметров ======================
    def update_params(self):
        for i in reversed(range(self.param_layout.rowCount())):
            self.param_layout.removeRow(i)
        self.param_entries.clear()

        func_name = self.func_combo.currentText()
        defaults = {
            'linear': {'a': '2.0', 'b': '1.0'},
            'quadratic': {'a': '1.0', 'b': '0.0', 'c': '0.0'},
            'cubic': {'a': '1.0', 'b': '0.0', 'c': '0.0', 'd': '0.0'},
            'sine_wave': {'A': '1.0', 'freq': '1.0', 'phase': '0.0'},
            'damped_cosine': {'A': '1.0', 'freq': '1.0', 'decay': '0.5', 'phase': '0.0'},
            'exponential': {'base': '2.71828', 'scale': '1.0'},
            'logarithm': {'base': '2.71828'},
            'gaussian': {'mu': '0.0', 'sigma': '1.0'},
            'sigmoid': {'k': '1.0', 'x0': '0.0'},
        }

        for param, default in defaults.get(func_name, {}).items():
            entry = QLineEdit(default)
            self.param_layout.addRow(f"{param}:", entry)
            self.param_entries[param] = entry

    # ====================== Получение функции ======================
    def get_func(self):
        func_name = self.func_combo.currentText()
        base = self.available_funcs[func_name]
        params = {}
        for name, entry in self.param_entries.items():
            try:
                params[name] = float(entry.text())
            except ValueError:
                QMessageBox.critical(self, "Ошибка", f"Неверное значение параметра: {name}")
                return None
        return partial(base, **params)

    # ====================== Обработчики кнопок ======================
    def run_uniform(self):
        f = self.get_func()
        if not f: return
        try:
            a = float(self.a_entry.text())
            b = float(self.b_entry.text())
            n = int(self.n_entry.text())
            nodes = np.linspace(a, b, n)
            newton_lagrange_spline_uniform(f, x_nodes=nodes, dif=300)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def run_chebyshev(self):
        f = self.get_func()
        if not f: return
        try:
            ch_deg = int(self.ch_entry.text())
            newton_lagrange_spline_chebyshev(f, ch_deg=ch_deg, left=-1, right=1, dif=300)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def run_error(self):
        f = self.get_func()
        if not f: return
        try:
            nmin = int(self.nmin_entry.text())
            nmax = int(self.nmax_entry.text())
            newton_lagrange_spline_error(f, n_min=nmin, n_max=nmax, dif=150)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def run_spline_compare(self):
        f = self.get_func()
        if not f: return
        try:
            a = float(self.a_entry.text())
            b = float(self.b_entry.text())
            n = int(self.n_entry.text())
            nodes = np.linspace(a, b, n)
            uniform_chebyshev_spline(f, x_nodes=nodes, dif=300)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))


# ====================== Запуск ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InterpolationGUI()
    window.show()
    sys.exit(app.exec())