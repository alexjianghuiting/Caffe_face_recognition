# -*- coding: utf-8 -*-
from PyQt4 import QtCore, QtGui
import cv2
import os

class Capture(QtCore.QThread):
    def __init__(self, device_id, width=640, height=360):
        super(Capture, self).__init__()
        self.device_id = device_id
        self.frame = None
        self.timer_frm = QtCore.QTimer()
        self.timer = QtCore.QTimer()
        self.width = width
        self.height = height
        self.FPS = 25 # frequency per second

        # if device_id is a string, it loads the video from filesys, otherwise it will open webcam
        self.cap = cv2.VideoCapture(self.device_id)

        self.PROCESS_INTERVAL = 0.001
        self.capturing = False
        self.timer_frm.stop()
        self.timer.stop()

    def run(self):
        self.timer_frm.timeout.connect(self.get_cv_frame)
        self.timer_frm.stop()
        self.timer.timeout.connect(self.send_frame)
        self.timer.stop()

    def __del__(self):
        self.cap.release()

    def startCapture(self):
        if not self.timer_frm.isActive():
            self.timer_frm.start(1000 / self.FPS)
            self.timer.start(1000 * self.PROCESS_INTERVAL)
        self.capturing = True

    def endCapture(self):
        self.capturing = False
        self.timer_frm.stop()
        self.timer.stop()

    def quitCapture(self):
        self.cap.release()
        QtCore.QCoreApplication.quit()

    def send_frame(self): # send frame to backend for processing
        self.emit(QtCore.SIGNAL('getFrame(PyQt_PyObject)'), self.frame)

    def get_cv_frame(self):
        if self.capturing:
            _, self.frame = self.cap.read()