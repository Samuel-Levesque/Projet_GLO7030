from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys


class Receipt(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = None
        self.load_ui()
        self.load_signals()

    def load_ui(self):
        self.ui = loadUi('PaintOnAPanel.ui')
        self.show()

    def load_signals(self):
        pass

app = QApplication(sys.argv)
receipt = Receipt()
sys.exit(app.exec_()) 