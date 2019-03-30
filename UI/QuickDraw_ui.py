"""
Quick, Draw UI
Inspired by https://www.codeproject.com/Articles/373463/Painting-on-a-Widget for creation of drawing widget
"""


import sys
from PyQt5 import QtGui, uic, QtWidgets, QtCore
from painter_widget import Painter, Colour3, Point, Shape, Shapes
import random
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas


class QuickDrawUI(QtWidgets.QDialog):
    Brush = True
    DrawingShapes = Shapes()
    IsPainting = False
    IsEraseing = False

    CurrentColour = Colour3(0,0,0)
    CurrentWidth = 8
    ShapeNum = 0
    IsMouseing = False
    PaintPanel = 0

    def __init__(self):
        super(QuickDrawUI, self).__init__()
        uic.loadUi('QuickDraw.ui', self)
        
        self.predict_image_size = 256
        
        self.PaintPanel = Painter(self)
        self.PaintPanel.close()
        self.DrawingFrame.insertWidget(0,self.PaintPanel)
        self.DrawingFrame.setCurrentWidget(self.PaintPanel)
        
        self.set_white_background(self.DrawingFrame)
        self.set_white_background(self.PaintPanel)
        
        self.GraphPredictions = Figure()
        self.GraphPredictionsCanvas = FigureCanvas(self.GraphPredictions)
        self.Predictions_Window.setWidget(self.GraphPredictionsCanvas)
        
        self.Establish_Connections()
        self.classes_dictionnary = {'airplane': 0, 
                                    'alarm_clock': 1, 
                                    'ambulance': 2, 
                                    'angel': 3}  # TODO Import all classes' names
        
        self.show()
        
        
    def SwitchBrush(self):
        if(self.Brush == True):
            self.Brush = False
        else:
            self.Brush = True
    
    def ChangeColour(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.CurrentColour = Colour3(col.red(),col.green(),col.blue())
   
    def ChangeThickness(self,num):
        self.CurrentWidth = num
            
    def ClearSlate(self):
        self.DrawingShapes = Shapes()
        self.PaintPanel.repaint()
        
    def PredictImage(self):
        pixmap = self.DrawingFrame.grab()
        pixmap_resized = pixmap.scaled(self.predict_image_size, self.predict_image_size)
        
        pixmap.save("image.png", "PNG")
        pixmap_resized.save("image_resized.png", "PNG")
        print("PNG Images saved in folder")


        # Random predictions graph       
        predictions = ["airplane", "angel", "ambulance"]  # TODO Use model predictions
        predictions_confidence = [0.5, 0.3, 0.2]  # TODO Use model predictions
        
        
        data = [random.random() for i in range(10)]
        ax = self.GraphPredictions.add_subplot(111)
        ax.clear()
        ax.bar(predictions, predictions_confidence)

        # refresh canvas
        self.GraphPredictionsCanvas.draw()
        
    
    def GenerateDrawingIdea(self):
        generated_drawing = random.choice(self.classes_dictionnary.keys())
        self.Generated_Drawing_TextEdit.setPlainText(generated_drawing)
              
    def Establish_Connections(self):
        self.BrushErase_Button.clicked.connect(self.SwitchBrush)
        self.ChangeColour_Button.clicked.connect(self.ChangeColour)
        self.Clear_Button.clicked.connect(self.ClearSlate)
        self.Predict_Button.clicked.connect(self.PredictImage)
        self.Drawing_Idea_Button.clicked.connect(self.GenerateDrawingIdea)
        self.Thickness_Spinner.valueChanged.connect(self.ChangeThickness)
        
    def set_white_background(self, widget):
        palette = widget.palette()
        palette.setColor(widget.backgroundRole(), QtCore.Qt.white)
        widget.setPalette(palette)
    


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = QuickDrawUI()
    sys.exit(app.exec_())
    
    
    