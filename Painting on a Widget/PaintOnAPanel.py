################################################
#              Paint on a Panel                #
#                                              #
#              By Geoff Samuel                 #
#            www.GeoffSamuel.com               #
#            Info@GeoffSamuel.com              #
################################################
#!!!!!! PLEASE DO NOT REMOVE THIS NOTICE !!!!!!#
################################################
#
#



# Load in PyQt and Maya libs
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

#Path to UI File
uifile = 'PaintOnAPanel.ui'
form, base = uic.loadUiType(uifile)

## My Own Colour Class, simple and light weight
class Colour3:
    R = 0
    G = 0
    B = 0
    #CONSTRUCTOR
    def __init__(self): 
        self.R = 0
        self.G = 0
        self.B = 0
    #CONSTRUCTOR - with the values to give it
    def __init__(self, nR, nG, nB):
        self.R = nR
        self.G = nG
        self.B = nB
        
## My Own Point Class, simple and light weight
class Point:
    #X Coordinate Value
    X = 0
    #Y Coordinate Value
    Y = 0
    #CONSTRUCTOR
    def __init__(self): 
        self.X = 0
        self.Y = 0
    #CONSTRUCTOR - with the values to give it
    def __init__(self, nX, nY):
        self.X = nX
        self.Y = nY
    #So we can set both values at the same time
    def Set(self,nX, nY):
        self.X = nX
        self.Y = nY 
        
        
## Shape class; holds data on the drawing point
class Shape:
    Location = Point(0,0)
    Width = 0.0
    Colour = Colour3(0,0,0)
    ShapeNumber = 0
    #CONSTRUCTOR - with the values to give it
    def __init__(self, L, W, C, S):
        self.Location = L
        self.Width = W
        self.Colour = C
        self.ShapeNumber = S


class Shapes:
    #Stores all the shapes
    __Shapes = []
    def __init__(self):
        self.__Shapes = []
    #Returns the number of shapes being stored.
    def NumberOfShapes(self):
        return len(self.__Shapes)
    #Add a shape to the database, recording its position,
    #width, colour and shape relation information
    def NewShape(self,L,W,C,S):
        Sh = Shape(L,W,C,S)
        self.__Shapes.append(Sh)
    #returns a shape of the requested data.
    def GetShape(self, Index):
        return self.__Shapes[Index]
    #Removes any point data within a certain threshold of a point.
    def RemoveShape(self, L, threshold):
        #do while so we can change the size of the list and it wont come back to bite me in the ass!!
        i = 0
        while True:
            if(i==len(self.__Shapes)):
                break 
            #Finds if a point is within a certain distance of the point to remove.
            if((abs(L.X - self.__Shapes[i].Location.X) < threshold) and (abs(L.Y - self.__Shapes[i].Location.Y) < threshold)):
                #removes all data for that number
                del self.__Shapes[i]
                #goes through the rest of the data and adds an extra
                #1 to defined them as a seprate shape and shuffles on the effect.
                for n in range(len(self.__Shapes)-i):
                    self.__Shapes[n+i].ShapeNumber += 1
                #Go back a step so we dont miss a point.
                i -= 1
            i += 1
                    

class Painter(QWidget):
    ParentLink = 0
    MouseLoc = Point(0,0)  
    LastPos = Point(0,0)  
    def __init__(self,parent):
        super(Painter, self).__init__()
        self.ParentLink = parent
        self.MouseLoc = Point(0,0)
        self.LastPos = Point(0,0) 
    #Mouse down event
    def mousePressEvent(self, event): 
        if(self.ParentLink.Brush == True):
            self.ParentLink.IsPainting = True
            self.ParentLink.ShapeNum += 1
            self.LastPos = Point(0,0)
        else:
            self.ParentLink.IsEraseing = True      
    #Mouse Move event        
    def mouseMoveEvent(self, event):
        if(self.ParentLink.IsPainting == True):
            self.MouseLoc = Point(event.x(),event.y())
            if((self.LastPos.X != self.MouseLoc.X) and (self.LastPos.Y != self.MouseLoc.Y)):
                self.LastPos =  Point(event.x(),event.y())
                self.ParentLink.DrawingShapes.NewShape(self.LastPos,self.ParentLink.CurrentWidth,self.ParentLink.CurrentColour,self.ParentLink.ShapeNum)
            self.repaint()
        if(self.ParentLink.IsEraseing == True):
            self.MouseLoc = Point(event.x(),event.y())
            self.ParentLink.DrawingShapes.RemoveShape(self.MouseLoc,10)     
            self.repaint()        
                
    #Mose Up Event         
    def mouseReleaseEvent(self, event):
        if(self.ParentLink.IsPainting == True):
            self.ParentLink.IsPainting = False
        if(self.ParentLink.IsEraseing == True):
            self.ParentLink.IsEraseing = False  
    
    def paintEvent(self,event):
        painter = QPainter()
        painter.begin(self)
        self.drawLines(event, painter)
        painter.end()
        
    def drawLines(self, event, painter):
        painter.setRenderHint(QPainter.Antialiasing);
        
        for i in range(self.ParentLink.DrawingShapes.NumberOfShapes()-1):
            
            T = self.ParentLink.DrawingShapes.GetShape(i)
            T1 = self.ParentLink.DrawingShapes.GetShape(i+1)
        
            if(T.ShapeNumber == T1.ShapeNumber):
                pen = QPen(QColor(T.Colour.R,T.Colour.G,T.Colour.B), T.Width/2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(T.Location.X,T.Location.Y,T1.Location.X,T1.Location.Y)
        
#Main UI Class
class CreateUI(base, form):
    Brush = True
    DrawingShapes = Shapes()
    IsPainting = False
    IsEraseing = False

    CurrentColour = Colour3(0,0,0)
    CurrentWidth = 10
    ShapeNum = 0
    IsMouseing = False
    PaintPanel = 0
    #Constructor
    def __init__(self):
        super(base,self).__init__()
        self.setupUi(self)
        self.setObjectName('Rig Helper')
        self.PaintPanel = Painter(self)
        self.PaintPanel.close()
        self.DrawingFrame.insertWidget(0,self.PaintPanel)
        self.DrawingFrame.setCurrentWidget(self.PaintPanel)
        self.Establish_Connections()
    
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
        


#New instance of the UI Form class, creating and showing it
def main():
    global PyForm
    PyForm=CreateUI()
    PyForm.show()
    
#check if another instance exsists.
if __name__=="__main__":
    main()

