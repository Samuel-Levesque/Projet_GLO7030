
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

from freehandTool.pointerEvent import PointerEvent
from freehandTool.freehand import FreehandTool
# from freehandTool.ghostLine import PointerTrackGhost
from freehandTool.freehandHead import PointerTrackGhost
from freehandTool.segmentString.segmentString import SegmentString
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

import random # TODO Remove after predict method is final


class DiagramScene(QGraphicsScene):
  def __init__(self, *args):
    QGraphicsScene.__init__(self, *args)
#    self.addItem(QGraphicsTextItem("Freehand drawing with pointer"))
    
    
    
class GraphicsView(QGraphicsView):
  def __init__(self, parent=None):
      super(GraphicsView, self).__init__(parent)
      
      assert self.dragMode() == QGraphicsView.NoDrag
      self.setRenderHint(QPainter.Antialiasing)
      self.setRenderHint(QPainter.TextAntialiasing)
      self.freehandTool = FreehandTool(view=self)
      self.setMouseTracking(True);  # Enable mouseMoveEvent
      



  ''' Delegate events to FreehandTool. '''
    
  def mouseMoveEvent(self, event):
    ''' Tell freehandTool to update its SegmentString. '''
    pointerEvent = PointerEvent()
    pointerEvent.makeFromEvent(mapper=self, event=event)
    self.freehandTool.pointerMoveEvent(pointerEvent)
  
  
  def mousePressEvent(self, event):
    '''
    On mouse button down, create a new (infinitesmal) SegmentString and PointerTrackGhost.
    freehandTool remembers and updates SegmentString.
    '''
    pointerEvent = PointerEvent()
    pointerEvent.makeFromEvent(mapper=self, event=event)
    
    '''
    freehandCurve as QGraphicsItem positioned at event in scene.
    It keeps its internal data in its local CS
    '''
    freehandCurve = SegmentString()
    self.scene().addItem(freehandCurve)
    freehandCurve.setPos(pointerEvent.scenePos)

    '''
    headGhost at (0,0) in scene
    it keeps its local data in CS equivalent to scene
    '''
    headGhost = PointerTrackGhost()
    self.scene().addItem(headGhost)
    
    self.freehandTool.setSegmentString(segmentString=freehandCurve, 
                                       pathHeadGhost=headGhost, 
                                       scenePosition=pointerEvent.scenePos)
    self.freehandTool.pointerPressEvent(pointerEvent)

    
  def mouseReleaseEvent(self, event):
    pointerEvent = PointerEvent()
    pointerEvent.makeFromEvent(mapper=self, event=event)
    self.freehandTool.pointerReleaseEvent(pointerEvent)
  
  
  def keyPressEvent(self, event):
    if event.modifiers() & Qt.ControlModifier:
      alternateMode = True
    else:
      alternateMode = False
    self.freehandTool.testControlPoint(event, alternateMode)
    
  
      
      

class MainWindow(QMainWindow):
    def __init__(self, *args):
        QMainWindow.__init__(self, *args)

        self.image_width = 800
        self.image_height = 800
        self.start_width = 100
        self.start_height = 100
        
        self.scene = DiagramScene()
        self.view = GraphicsView(self.scene)
        rect = QRectF(self.start_width, self.start_height, self.image_width, self.image_height)
        self.view.fitInView(rect)
        self.view.setSceneRect(rect)
        self.setCentralWidget(self.view)

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
        outputimg = QPixmap(self.image_width, self.image_height)
        QPixmapCache.clear()
        
        painter = QPainter(outputimg)
        painter.setRenderHint(QPainter.Antialiasing)
        targetrect = QRectF(self.start_width, self.start_height, self.image_width, self.image_height)
        sourcerect = QRect(self.start_width, self.start_height, self.image_width, self.image_height)
        self.view.render(painter, targetrect, sourcerect)
        outputimg.save("test.png", "PNG")
        
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
        
        
        
def main(args):
    app = QApplication(args)
    app.setStyle(QStyleFactory.create("Fusion"))  # fixes gtk assertion errors
    mainWindow = MainWindow()
    mainWindow.setGeometry(100, 100, 1000, 1000)
    mainWindow.show()

    sys.exit(app.exec_()) # Qt Main loop


if __name__ == "__main__":
    main(sys.argv)
