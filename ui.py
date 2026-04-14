import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QGroupBox,
    QFormLayout, QMessageBox
)
from core.funcs import f1, f2, f3
from core.plots import spline_plot, error_plot


class SplineGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Кубический сплайн интерполяция")
        self.setGeometry(100, 100, 500, 400)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(15)

        # ---------- Выбор функции ----------
        func_group = QGroupBox("Выберите функцию")
        func_layout = QVBoxLayout()
        func_group.setLayout(func_layout)

        self.func_combo = QComboBox()
        self.available_funcs = {
            'f1(x) = sin(5x)': f1,
            'f2(x) = e^{-x^2}': f2,
            'f3(x) = arctan(10x)': f3
        }
        self.func_combo.addItems(self.available_funcs.keys())
        func_layout.addWidget(self.func_combo)

        # ---------- Параметры ----------
        params_group = QGroupBox("Параметры")
        params_layout = QFormLayout()
        params_group.setLayout(params_layout)

        self.n_entry = QLineEdit("10")
        params_layout.addRow("Число узлов n:", self.n_entry)

        info_label = QLabel("Интервал фиксирован: [-1, 1]")
        info_label.setStyleSheet("color: gray;")
        params_layout.addRow(info_label)

        # ---------- Кнопки ----------
        btn_layout = QVBoxLayout()
        btn_style = "QPushButton { padding: 12px; font-size: 14px; font-weight: bold; }"

        self.btn_spline = QPushButton("Построить сплайн (равномерные узлы)")
        self.btn_error = QPushButton("Построить график ошибки от n")

        for btn in (self.btn_spline, self.btn_error):
            btn.setStyleSheet(btn_style)
            btn_layout.addWidget(btn)

        self.btn_spline.clicked.connect(self.run_spline)
        self.btn_error.clicked.connect(self.run_error)

        # ---------- Сборка ----------
        main_layout.addWidget(func_group)
        main_layout.addWidget(params_group)
        main_layout.addLayout(btn_layout)
        main_layout.addStretch()

    def get_func(self):
        """Возвращает выбранную функцию."""
        func_name = self.func_combo.currentText()
        return self.available_funcs[func_name]

    def run_spline(self):
        """Обработчик кнопки построения сплайна."""
        func = self.get_func()
        try:
            n = int(self.n_entry.text())
            if n < 3:
                QMessageBox.warning(self, "Предупреждение",
                                    "Число узлов должно быть не менее 3.")
                return
            spline_plot(n, func)
        except ValueError:
            QMessageBox.critical(self, "Ошибка",
                                 "Неверное значение n. Введите целое число.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def run_error(self):
        """Обработчик кнопки построения графика ошибки."""
        func = self.get_func()
        try:
            error_plot(func)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SplineGUI()
    window.show()
    sys.exit(app.exec())