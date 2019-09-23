# -*- coding: utf-8 -*-
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import QObject
import main_window

class MyGui(QtGui.QMainWindow, main_window.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyGui,self).__init__(parent)
        self.setupUi(self)

        QObject.connect(self.dial, QtCore.SIGNAL("valueChanged(int)"), self.lcdNumber.display)

        self.checkBox_2.stateChanged.connect(lambda:self.check1_state(self.checkBox, self.checkBox_3))
        self.frame = None
        self.painter = QtGui.QPainter()
        self.gender_pred = [[],[]]
        self.gender_label = ['Female', 'Male']

    def check1_state(self, check2, check3):
        if not self.checkBox_2.isChecked():
            check2.setChecked(False)
            check3.setChecked(False)

    def setFrame(self, frame):
        self.frame = frame
        pixmap = QtGui.QPixmap.fromImage(self.frame)
        scaledpixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(scaledpixmap)

    def drawFace(self, face_info):
        # face_info, from faceDetector
        # self.face_info[k] = ([d.left(), d.top(), d.right(), d.bottom()], landmarks[18:], crop_face)
        self.frame = face_info[1]
        face_info = face_info[0]

        image = QtGui.QImage(self.frame.tostring(), self.frame.shape[1], self.frame.shape[0],
                             QtGui.QImage.Format_RGB888).rgbSwapped()

        self.painter.begin(image)
        self.painter.setRenderHint(QtGui.QPainter.Antialiasing)
        self.painter.setBrush(QtGui.QBrush(QtGui.QColor(192, 192, 255)))

        for id, face in face_info.items():
            # every face(auto-iter), no need for id
            x, y = face[0][0], face[0][1]
            w = face[0][2] - x
            h = face[0][3] - y
            self.drawLines(x, y, w, h)
            self.drawPoints(face[1])

        g_str = self.gender_pred[0]
        g_pos = self.gender_pred[1]
        for i in range(len(g_str)):
            font = self.painter.font()
            font.setPixelSize(20)
            self.painter.setFont(font)
            self.painter.drawText(int(g_pos[i][0]), int(g_pos[i][1]), QtCore.QString(self.gender_label[g_str[i]]))

        self.gender_pred = [[], []]
        self.painter.end()

        pixmap = QtGui.QPixmap.fromImage(image)
        scaledpixmap = pixmap.scaled(self.label.size())
        # pixmap, a kind of pain device
        self.label.setPixmap(scaledpixmap)

    def drawLines(self, x, y, w, h):
        pen = QtGui.Qpen(QtCore.Qt.green, 5, QtCore.Qt.SolidLine)
        sz = 0.1
        self.painter.setPen(pen)
        # draw a line from (x1,y1) to (x2,y2)
        self.painter.drawLine(x, y, x + w * sz, y)
        self.painter.drawLine(x, y, x, y + h * sz)

        self.painter.drawLine(x + w, y, x + w, y + h * sz)
        self.painter.drawLine(x + w, y, x + w * (1 - sz), y)

        self.painter.drawLine(x, y + h, x, y + h * (1 - sz))
        self.painter.drawLine(x, y + h, x + w * sz, y + h)

        self.painter.drawLine(x + w, y + h, x + w * (1 - sz), y + h)
        self.painter.drawLine(x + w, y + h, x + w, y + h * (1 - sz))

        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtCore.Qt.blue)

        self.painter.setBrush(brush)
        brush.setStyle(QtCore.Qt.NoBrush)
        self.painter.setBrush(brush)

        pen = QtGui.QPen(QtCore.Qt.darkBlue, 2, QtCore.Qt.DotLine)
        self.painter.setPen(pen)
        self.painter.drawRect(x, y, w, h)

    def drawPoints(self, landmarks):
        pen = QtGui.QPen(QtCore.Qt.darkBlue, 4, QtCore.Qt.SolidLine)
        self.painter.setPen(pen)
        for m in landmarks:
            self.painter.drawPoint(m[0], m[1])

    def drawGender(self, face_info):
        self.gender_pred = face_info

