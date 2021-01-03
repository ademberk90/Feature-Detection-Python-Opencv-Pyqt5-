from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from baykar_python import Ui_MainWindow
from PyQt5 import QtGui
import os
import cv2
import numpy as np

class mainApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        """size of image"""
        self.disply_width = 1800
        self.display_height = 1300
        """orb standart parameters except fast threshold"""
        self.nfeatures = 500
        self.scaleFactor = 1.2
        self.nlevels = 8
        self.edgeThreshold = 31
        self.firstLevel = 0
        self.WTA_K = 2
        self.patchSize = 31
        self.fastThreshold = 0

        self.img1_set = 0
        self.img2_set = 0
        """set sliders tracking to false until image uploaded"""
        self.ui.horizontalSlider.setTracking(False)
        self.ui.horizontalSlider_2.setTracking(False)
        self.ui.horizontalSlider_3.setTracking(False)
        self.ui.horizontalSlider_4.setTracking(False)
        self.ui.horizontalSlider_5.setTracking(False)
        self.ui.horizontalSlider_6.setTracking(False)
        self.ui.horizontalSlider_7.setTracking(False)
        self.ui.horizontalSlider_8.setTracking(False)
        """connect the slider to function when value changed"""
        self.ui.horizontalSlider.valueChanged[int].connect(self.changeNfeatures)
        self.ui.horizontalSlider_2.valueChanged[int].connect(self.changeEdgeThr)
        self.ui.horizontalSlider_3.valueChanged[int].connect(self.changePatchSize)
        self.ui.horizontalSlider_4.valueChanged[int].connect(self.changeScale)
        self.ui.horizontalSlider_5.valueChanged[int].connect(self.changeFirstLev)
        self.ui.horizontalSlider_6.valueChanged[int].connect(self.changeFastThr)
        self.ui.horizontalSlider_7.valueChanged[int].connect(self.changeNlevels)
        self.ui.horizontalSlider_8.valueChanged[int].connect(self.changeWTA)
        """connect the button to function when press them and open file dialog"""
        self.ui.pushButton_2.clicked.connect(self.takeQueryImg)
        self.ui.pushButton.clicked.connect(self.takeTrainImg)

    def getEnable(self):
        """if image uploaded set enable the sliders"""
        if self.img1_set == 1 and self.img2_set == 1:
            self.ui.horizontalSlider.setEnabled(True)
            self.ui.horizontalSlider_2.setEnabled(True)
            self.ui.horizontalSlider_3.setEnabled(True)
            self.ui.horizontalSlider_4.setEnabled(True)
            self.ui.horizontalSlider_5.setEnabled(True)
            self.ui.horizontalSlider_6.setEnabled(True)
            self.ui.horizontalSlider_7.setEnabled(True)
            self.ui.horizontalSlider_8.setEnabled(True)
    def takeQueryImg(self):
        """take the filename and read it """
        filename = QFileDialog.getOpenFileName()
        queryPath = filename[0]
        self.img1_set = True
        self.img1 = cv2.imread(queryPath)
        self.getEnable()

    def takeTrainImg(self):
        """take the filename and read it """
        filename = QFileDialog.getOpenFileName()
        trainPath = filename[0]
        self.img2_set = True
        self.img2 = cv2.imread(trainPath)
        self.getEnable()

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def changeNfeatures(self,value):
        self.nfeatures = value
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures, scaleFactor=self.scaleFactor, nlevels=self.nlevels, edgeThreshold=self.edgeThreshold,
                             firstLevel=self.firstLevel, WTA_K=self.WTA_K,
                             patchSize=self.patchSize, fastThreshold=self.fastThreshold)

        self.detAndComp(self.orb)

    def changeScale(self,value):
        if value == 10:
            self.scaleFactor = 1.1
        elif value == 20:
            self.scaleFactor = 1.9
        else:
            self.scaleFactor = value/10

        self.orb = cv2.ORB_create(nfeatures=self.nfeatures, scaleFactor=self.scaleFactor, nlevels=self.nlevels,
                             edgeThreshold=self.edgeThreshold,
                             firstLevel=self.firstLevel, WTA_K=self.WTA_K,
                             patchSize=self.patchSize, fastThreshold=self.fastThreshold)
        self.detAndComp(self.orb)

    def changeNlevels(self,value):
        self.nlevels = value
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures, scaleFactor=self.scaleFactor, nlevels=self.nlevels,
                             edgeThreshold=self.edgeThreshold,
                             firstLevel=self.firstLevel, WTA_K=self.WTA_K,
                             patchSize=self.patchSize, fastThreshold=self.fastThreshold)
        self.detAndComp(self.orb)

    def changeEdgeThr(self,value):
        self.edgeThreshold = value
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures, scaleFactor=self.scaleFactor, nlevels=self.nlevels,
                             edgeThreshold=self.edgeThreshold,
                             firstLevel=self.firstLevel, WTA_K=self.WTA_K,
                             patchSize=self.patchSize, fastThreshold=self.fastThreshold)
        self.detAndComp(self.orb)

    def changeFirstLev(self,value):
        self.firstLevel = value
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures, scaleFactor=self.scaleFactor, nlevels=self.nlevels,
                             edgeThreshold=self.edgeThreshold,
                             firstLevel=self.firstLevel, WTA_K=self.WTA_K,
                             patchSize=self.patchSize, fastThreshold=self.fastThreshold)
        self.detAndComp(self.orb)

    def changeWTA(self,value):
        self.WTA_K = value
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures, scaleFactor=self.scaleFactor, nlevels=self.nlevels,
                             edgeThreshold=self.edgeThreshold,
                             firstLevel=self.firstLevel, WTA_K=self.WTA_K,
                             patchSize=self.patchSize, fastThreshold=self.fastThreshold)
        self.detAndComp(self.orb)

    def changePatchSize(self,value):
        self.patchSize = value
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures, scaleFactor=self.scaleFactor, nlevels=self.nlevels,
                             edgeThreshold=self.edgeThreshold,
                             firstLevel=self.firstLevel, WTA_K=self.WTA_K,
                             patchSize=self.patchSize, fastThreshold=self.fastThreshold)
        self.detAndComp(self.orb)

    def changeFastThr(self,value):
        self.fastThreshold = value
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures, scaleFactor=self.scaleFactor, nlevels=self.nlevels,
                             edgeThreshold=self.edgeThreshold,
                             firstLevel=self.firstLevel, WTA_K=self.WTA_K,
                             patchSize=self.patchSize, fastThreshold=self.fastThreshold)
        self.detAndComp(self.orb)

    def detAndComp(self,orb):
        """find keypoints and descriptors"""
        kp1, des1 = orb.detectAndCompute(self.img1, None)
        kp2, des2 = orb.detectAndCompute(self.img2, None)
        """initiliaze the matcher"""
        matcher = cv2.BFMatcher()
        matches_all = matcher.match(des1, des2)
        matches_gms = cv2.xfeatures2d.matchGMS(self.img1.shape[:2], self.img2.shape[:2], kp1, kp2, matches_all, withScale=True, withRotation=True, thresholdFactor=6)
        matches_gms = sorted(matches_gms, key = lambda x:x.distance)
        """set the maximum des. will shown in figure"""
        max = 200
        if len(matches_gms) < 200:
            max = len(matches_gms)
        """these comment lines for the finding the area but it can be wrong """
        # good = []
        # for m, n in matches:
        #     if m.distance < 0.7 * n.distance:
        #         good.append(m)

        # if len(good) > 5:
        #     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        #     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        #
        #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        #     # M=np.linalg.pinv(M)
        #     matchesMask = mask.ravel().tolist()
        #
        #     h, w = self.img1.shape
        #
        #     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        #     dst = cv2.perspectiveTransform(pts, M)
        """draw the matches"""
        img3 = cv2.drawMatches(self.img1, kp1, self.img2, kp2, matches_gms[:max], None, matchColor=(255,0,0),flags=2)
        """add the image to label as pixmap"""
        qt_img = self.convert_cv_qt(img3)
        self.ui.label.setPixmap(qt_img)

app = QApplication([])
window = mainApp()
window.show()
app.exec_()
