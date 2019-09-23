# -*- coding: utf-8 -*-
import dlib
from PyQt4 import QtCore
import numpy as np

class Face_detector(QtCore.QThread):
    def __init__(self, textBrowser):
        super(Face_detector,self).__init__()
        self.face_detector = dlib.get_frontal_face_detector()
        self.ldmark_detector = dlib.shape_predictor('./dlib_model/shape_predictor_68_face_landmarks.dat')
        self.face_info = {}
        self.textBrowser = textBrowser
        self.detecting = True
        self.ldmarking = False
        self.total = 0

    # The bigger the score, the more confident each prediction is

    def detect_face(self, img):
        if self.detecting:
            self.face_info = {}
            # 1 indicates that we unsample the image 1 time, making everything bigger and allowing us to detect more faces
            dets = self.face_detector(img,0)

            if len(dets) > 0:
                self.textBrowser.append('Number of face detected: {}'.format(len(dets)))

            for k,d in enumerate(dets):
                landmarks = []
                if self.ldmarking:
                    shape = self.ldmark_detector(img, d)
                    landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            crop_face = np.copy(img[max(0, d.top()):d.bottom(), max(0, d.left()):d.right(), :])
            self.total += 1
            self.face_info[k] = ([d.left(), d.top(), d.right(), d.bottom(), landmarks[18:], crop_face])

        self.emit(QtCore.SIGNAL('det(PyQt_PyObject'), [self.face_info, img])

    def onOffdet(self, checkbox):
        if checkbox.isChecked():
            self.detecting = True
        else:
            self.detecting = False
    def onOffldmark(self, checkbox):
        if checkbox.isChecked():
            self.ldmarking = True
        else:
            self.ldmarking = False