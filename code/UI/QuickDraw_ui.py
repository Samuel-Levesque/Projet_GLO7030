"""
Quick, Draw UI
Inspired by https://www.codeproject.com/Articles/373463/Painting-on-a-Widget for creation of drawing widget
"""


import sys
from PyQt5 import QtGui, uic, QtWidgets, QtCore
from UI.painter_widget import Painter, Colour3, Point, Shape, Shapes
import random
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from data_set_from_image import predict_image_classes
from utility import load_object
import pickle
import cv2
import PyQt5 as Qt


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
        uic.loadUi('UI/QuickDraw.ui', self)

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

        with open("saves_obj/full_encoding_dict.pickle", 'rb') as handle:
            self.classes_dictionnary = pickle.load(handle)

        with open("saves_obj/full_decoding_dict.pickle", 'rb') as handle:
            self.decoding_dict = pickle.load(handle)

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
        pixmap.save("temp_image/pixmap_image.png", "PNG")
        opencv_image = cv2.imread("temp_image/pixmap_image.png", 0)
        centered_cv_image = self.center_cv_image(opencv_image)
        centered_pixmap = QtGui.QPixmap(r"temp_image/centered_image.png")
        pixmap_resized = centered_pixmap.scaled(self.predict_image_size, self.predict_image_size, aspectRatioMode=QtCore.Qt.KeepAspectRatio)

        pixmap_resized.save("UI/image/image_resized.png", "PNG")
        print(pixmap_resized.size)
        print("PNG Images saved in folder")

        predictions, predictions_confidence = predict_image_classes(path_image_folder="UI/",
                                                                    path_save_model="modele/model_poids_mauvaise_classes_sampling.tar",
                                                                    use_gpu=False,
                                                                    decoding_dict=self.decoding_dict,
                                                                    file_number=1)

        ax = self.GraphPredictions.add_subplot(111)
        ax.clear()
        ax.bar(predictions, predictions_confidence)

        # refresh canvas
        self.GraphPredictionsCanvas.draw()


    def GenerateDrawingIdea(self):
        generated_drawing = random.choice(list(self.classes_dictionnary.keys()))
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

    def center_cv_image(self, opencv_image):

        th, threshed = cv2.threshold(opencv_image, 240, 255, cv2.THRESH_BINARY_INV)

        ## (2) Morph-op to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

        ## (3) Find the max-area contour
        cnts = cv2.findContours(morphed[1:-1, 1:-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv2.contourArea)[-1]

        ## (4) Crop and save it
        x,y,w,h = cv2.boundingRect(cnt)
        y += 1
        x += 1
        dst = opencv_image[y:y+h, x:x+w]

        cv2.imwrite("temp_image/centered_image.png", dst)



if __name__ == '__main__':
    import os
    os.chdir('/Users/Samuel_Levesque/Documents/GitHub/Projet_GLO7030/code')
#    os.chdir("D:/User/William/Documents/Devoir/Projet Deep/Projet_GLO7030/code")

    app = QtWidgets.QApplication(sys.argv)
    window = QuickDrawUI()
    sys.exit(app.exec_())


