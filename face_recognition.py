# -*- coding: utf-8 -*-
from PyQt4 import QtCore
from caffe_net import *
import glob
import caffe
import sklearn.metrics.pairwise
import cv2

class Face_recognizor(QtCore.QThread):
    def __init__(self, textBrowser):
        super(Face_recognizor, self).__init__()

        # load face model
        caffemodel = './deep_model/VGG_FACE.caffemodel'
        deploy_file = './deep_model/VGG_FACE_deploy.prototxt'
        mean_file = None
        self.net = Model_net(caffemodel, deploy_file, gpu=True)

        self.recognizing = True
        self.textBrowser = textBrowser
        self.threshold = 0
        self.label = ['Stranger']
        self.db_path = './db'
        self.db = None
        self.load_db()

    def load_db(self):
        if not os.path.exists(self.db_path):
            print ('db path does not exist')
        folders = sorted(glob.glob(os.path.join(self.db_path, '*')))
        for name in folders:

            print('loading {}:'.format(name))
            self.label.append(os.path.basename(name))
            img_list = glob.glob(os.path.join(name, '*.jpg'))

            imgs = [cv2.imread(img) for img in img_list]
            scores, pred_labels, fea = self.net.classify(imgs, layer_name='fc7')

            fea = np.mean(fea, 0)
            print (fea[:])
            if self.db is None:
                self.db = fea.copy()
            else:
                self.db = np.vstack(self.db, fea.copy())

            print('done')
        print (self.label)

    def face_recognition(self, face_info):
        if self.recognizing:
            img = []
            cord = []
            for k, face in face_info[0].items():
                face_norm = face[2].astype(float)
                face_norm = cv2.resize(face_norm, (128, 128))
                img.append(face_norm)
                cord.append(face[0][0:2])

            if len(img) != 0:
                prob, pred, fea = self.net.classify(img, layer_name='fc7')

                dist = sklearn.metrics.pairwise.cosine_similarity(fea, self.db)
                pred = np.argmax(dist,1)
                dist = np.max(dist,1)

                pred = [0 if dist[i] <= self.threshold/100.0 else pred[i]+1 for i in range(len(dist))]

                msg = QtCore.QString("Face Recognition Pred: ".format(' '.join(self.label[x] for x in pred())))
                self.textBrowser.append(msg)
                self.emit(QtCore.SIGNAL('face_id(PyQt_PyObject)'), [pred, cord])

    def set_threshold(self, th):
        self.threshold = th
        self.textBrowser.append('Threshold has been changed to: {}'.format(self.threshold))

    def OnOffFaceRecognizor(self, checkbox):
        if checkbox.isChecked():
            self.recognizing = True
        else:
            self.recognizing = False