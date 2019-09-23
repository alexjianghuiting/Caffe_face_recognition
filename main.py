# -*- coding: utf-8 -*-
from PySide.QtWebKit import *

import sys
from MyGui import *
from capture import  *
from faceDetector import *
from gender_recognition import *
from face_recognition import *
from functools import partial

#PyQt function does not have ()
def main():
    app = QtGui.QApplication(['Face_Demo'])

    form = MyGui()

    # video capture thread
    capture = Capture(0)
    capture.start()
    form.pushButton.clicked.connect(capture.quitCapture)
    form.pushButton_2.clicked.connect(capture.startCapture)
    form.pushButton_3.clicked.connect(capture.endCapture)

    # face detector thread
    face_detector = Face_detector(form.textBrowser)
    face_detector.connect(capture, QtCore.SIGNAL("getFrame(PyQt_PyObject)"), face_detector.detect_face)

    # connect GUI widgets
    enable_slot_det = partial(face_detector.onOffdet, form.checkBox_2)
    form.checkBox_2.stateChanged.connect(lambda  x: enable_slot_det())

    enable_slot_ldmark = partial(face_detector.onOffldmark, form.checkBox)
    form.checkBox.stateChanged.connect(lambda  x: enable_slot_ldmark())
    form.connect(face_detector, QtCore.SIGNAL('det(PyQt_PyObject)'), form.drawFace)

    # create net for gender recognition
    gender_net = Gender_recognizor(form.textBrowser)
    gender_net.connect(face_detector, QtCore.SIGNAL('det(PyQt_PyObject)'), gender_net.gender_recognition)

    # connect to widgets
    enable_slot_gender = partial(gender_net.onOffgender, form.checkBox_3)
    form.checkBox_3.stateChanged.connect(lambda  x: enable_slot_gender())
    form.connect(gender_net, QtCore.SIGNAL('gender(PyQt_PyObject'), form.drawGender)

    # model net for face recognition
    form.dial.setValue(55)
    face_net = Face_recognizor(form.textBrowser)
    face_net.connect(face_detector, QtCore.SIGNAL('det(PyQt_PyObject'), face_net.face_recognition)

    form.dial.valueChanged.connect(face_net.set_threshold) # threshold changes, dial value changes

    form.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()