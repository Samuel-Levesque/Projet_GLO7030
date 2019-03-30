"""
Quick, Draw UI
Inspired by https://www.codeproject.com/Articles/373463/Painting-on-a-Widget for creation of drawing widget
"""


import sys
from PyQt5 import QtGui, uic, QtWidgets


class QuickDrawUI(QtWidgets.QDialog):
    def __init__(self):
        super(QuickDrawUI, self).__init__()
        uic.loadUi('QuickDraw.ui', self)
        self.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = QuickDrawUI()
    sys.exit(app.exec_())
    
    
    