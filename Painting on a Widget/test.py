from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import *

uifile = 'PaintOnAPanel.ui'
form, base = loadUiType(uifile)


class CreateUI(base, form):
    def __init__(self):
        super(base,self).__init__()
        self.setupUi(self)
        
def main():
    global PyForm
    PyForm=CreateUI()
    PyForm.show()
    
if __name__=="__main__":
    main()