
'''
Quick, Draw homemade GUI to show our model

Copyright 2012 Lloyd Konneker for Drawing framework

This is free software, covered by the GNU General Public License.
'''     

try:
    # Set PyQt API version to 2
    import sip
    API_NAMES = ["QDate", "QDateTime", "QString", "QTextStream", "QTime", "QUrl", "QVariant"]
    API_VERSION = 2
    for name in API_NAMES:
      sip.setapi(name, API_VERSION)

    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
except ImportError:
    from PySide.QtCore import *
    from PySide.QtGui import *

import sys

#from freehandTool.pointerEvent import PointerEvent
#from freehandTool.freehand import FreehandTool
# from freehandTool.ghostLine import PointerTrackGhost
#from freehandTool.freehandHead import PointerTrackGhost
#from freehandTool.segmentString.segmentString import SegmentString
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

import random # TODO Remove after predict method is final

###
from PaintOnAPanel import Painter, Shape, Shapes, Colour3, base, form
      

class MainWindow(QMainWindow):
    Brush = True
    DrawingShapes = Shapes()
    IsPainting = False
    IsEraseing = False

    CurrentColour = Colour3(0,0,0)
    CurrentWidth = 10
    ShapeNum = 0
    IsMouseing = False
    PaintPanel = 0
    
    def __init__(self, *args):
        QMainWindow.__init__(self, *args)
        
#        super(base,self).__init__()
        self.setupUi(self)
        self.setObjectName('Rig Helper')
        self.PaintPanel = Painter(self)
        self.PaintPanel.close()
        self.DrawingFrame.insertWidget(0,self.PaintPanel)
        self.DrawingFrame.setCurrentWidget(self.PaintPanel)
        self.Establish_Connections()
        

        self.image_width = 700
        self.image_height = 700
        self.start_width = 0
        self.start_height = 0
        
#        self.scene = DiagramScene()
#        self.view = GraphicsView(self.scene)
#        rect = QRectF(self.start_width, self.start_height, self.image_width, self.image_height)
#        self.view.fitInView(rect)
#        self.view.setSceneRect(rect)
        self.Painter = Painter(self)
        self.setCentralWidget(self.Painter)

        # Clear Button Definition
        self.ClearButtonDock = QDockWidget(self)
        self.ClearButtonWidget = QPushButton("Clear")
        self.ClearButtonDock.setWidget(self.ClearButtonWidget)
        self.ClearButtonDock.setFeatures(QDockWidget.DockWidgetVerticalTitleBar)
        
        self.ClearButtonWidget.clicked.connect(self.clear_drawing)
        
        # Predict Button Definition
        self.PredictButtonDock = QDockWidget(self)
        self.PredictButtonWidget = QPushButton("Predict")
        self.PredictButtonDock.setWidget(self.PredictButtonWidget)
        self.PredictButtonDock.setFeatures(QDockWidget.DockWidgetVerticalTitleBar)
        
        self.PredictButtonWidget.clicked.connect(self.predict_drawing)
        
        # Predictions graph definition
        self.GraphPredictions = Figure()
        self.GraphPredictionsCanvas = FigureCanvas(self.GraphPredictions)
        self.GraphPredictionsDock = QDockWidget(self)
        self.GraphPredictionsDock.setWidget(self.GraphPredictionsCanvas)
        self.GraphPredictionsDock.setFeatures(QDockWidget.DockWidgetVerticalTitleBar)
        
        # Adding widgets to main window
        self.addDockWidget(Qt.RightDockWidgetArea, self.ClearButtonDock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.PredictButtonDock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.GraphPredictionsDock)
        
    def predict_drawing(self):
        # TODO Do forward pass and generate predictions graph
        # Image extraction from drawing window
#        outputimg = QPixmap(self.image_width, self.image_height)
        outputimg = QPixmap(round(self.view.width()), round(self.view.width()))
#        QPixmapCache.clear()
        
        painter = QPainter(outputimg)
        painter.setRenderHint(QPainter.Antialiasing)
        targetrect = self.scene.sceneRect()
#        targetrect = QRectF(self.start_width, self.start_height, self.image_width, self.image_height)
#        sourcerect = QRect(self.start_width, self.start_height, self.image_width, self.image_height)
#        self.view.render(painter, targetrect, sourcerect)
        self.view.render(painter, targetrect)
        outputimg.save("test.png", "PNG")
        
#        test_pixmap = QPixmap(self.scene.width(), self.scene.width())
#        test_pixmap.grabWidget(self.scene)
        
        painter.end()
        
        # random data
        data = [random.random() for i in range(10)]

        # create an axis
        ax = self.GraphPredictions.add_subplot(111)

        # discards the old graph
        ax.clear()

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.GraphPredictionsCanvas.draw()
        
    def clear_drawing(self):
        self.scene.clear()
        
    def SwitchBrush(self):
        if(self.Brush == True):
            self.Brush = False
        else:
            self.Brush = True
    
    def ChangeColour(self):
        col = QColorDialog.getColor()
        if col.isValid():
            self.CurrentColour = Colour3(col.red(),col.green(),col.blue())
   
    def ChangeThickness(self,num):
        self.CurrentWidth = num
            
    def ClearSlate(self):
        self.DrawingShapes = Shapes()
        self.PaintPanel.repaint()  
        
              
    def Establish_Connections(self):
        self.BrushErase_Button.clicked.connect(self.SwitchBrush)
        self.ChangeColour_Button.clicked.connect(self.ChangeColour)
        self.Clear_Button.clicked.connect(self.ClearSlate)
        self.Thickness_Spinner.valueChanged.connect(self.ChangeThickness)
        
        
        
def main(args):
    app = QApplication(args)
    app.setStyle(QStyleFactory.create("Fusion"))  # fixes gtk assertion errors
    mainWindow = MainWindow()
    mainWindow.setGeometry(100, 100, 800, 1200)
    mainWindow.show()

    sys.exit(app.exec_()) # Qt Main loop


if __name__ == "__main__":
    main(sys.argv)
