import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QWidget
from PyQt5.QtGui import QIcon
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class SinPlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        
        # Main widget
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        
        # Widgets
        self.start_label = QLabel('Start:', main_widget)
        self.start_label.move(10, 10)
        self.start_line_edit = QLineEdit(main_widget)
        self.start_line_edit.move(60, 10)

        self.end_label = QLabel('End:', main_widget)
        self.end_label.move(10, 40)
        self.end_line_edit = QLineEdit(main_widget)
        self.end_line_edit.move(60, 40)
        
        self.btn = QPushButton('Plot', main_widget)
        self.btn.move(10, 70)
        self.btn.clicked.connect(self.plot)

        # Canvas for plot with absolute positioning
        self.canvas = FigureCanvas(Figure(figsize=(10, 8)))
        self.canvas.move(10, 100)
        self.canvas.setParent(main_widget)

        self.setGeometry(100, 100, 900, 800)  # Adjusted for larger canvas
        self.setWindowTitle('Sin plot in GUI with QMainWindow')

    def plot(self):
        start = float(self.start_line_edit.text())
        end = float(self.end_line_edit.text())
        x = np.linspace(start, end, 1000)
        y = np.sin(x)

        # Clear previous plot and plot the new data
        self.canvas.figure.clf()
        ax = self.canvas.figure.subplots()
        ax.plot(x, y)
        ax.set_title('Sine function')
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    ex = SinPlot()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()