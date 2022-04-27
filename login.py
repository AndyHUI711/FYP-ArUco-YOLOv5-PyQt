##Copyright
##WrittenModified by HUI, CHEUNG YUEN
##Student of HKUST
##FYP, FINAL YEAR PROJECT
# -*- coding: utf-8 -*-
#reference1 https://github.com/ultralytics/yolov5
#reference2 https://github.com/BonesCat/YoloV5_PyQt5/tree/main
#reference3 https://github.com/Javacr/PyQt5-YOLOv5
import base64
import math
from datetime import datetime
from PIL import ImageEnhance, Image
from utils.id_utils import get_id_info, sava_id_info

# login UI
from loginui import Ui_Login_Ui_Form
from cv2 import aruco
import ml_scale_use
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from UI import Ui_MainWindow
from signupui import Ui_Dialog

from pyzbar import pyzbar
import datetime
from lib.share import shareInfo

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from sklearn.ensemble import IsolationForest
import sys
import json
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression #using linearregression
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets #using RANSAC
import pickle
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox

from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device
from utils.capnums import Camera
from dialog.rtsp_win import Window

# Login interface
class win_Login(QMainWindow):
    def __init__(self, parent = None):
        super(win_Login, self).__init__(parent)
        self.ui_login = Ui_Login_Ui_Form()
        self.ui_login.setupUi(self)
        self.init_slots()
        self.hidden_pwd()
    # hide password
    def hidden_pwd(self):
        self.ui_login.lineEdit_2.setEchoMode(QLineEdit.Password)
    # init
    def init_slots(self):
        self.ui_login.pushButton.clicked.connect(self.onSignIn)
        self.ui_login.lineEdit_2.returnPressed.connect(self.onSignIn) # enter
        self.ui_login.pushButton_2.clicked.connect(self.create_id)
    # Jump to sign up interface
    def create_id(self):
        print("jump to signup")
        shareInfo.createWin = win_Register()
        shareInfo.createWin.show()
    # Save login log
    def sava_login_log(self, username):
        with open('login_log.txt', 'a', encoding='utf-8') as f:
            f.write(username + '\t log in at' + datetime.now().strftimestrftime+ '\r')
    # login
    def onSignIn(self):
        print("You pressed sign in")
        # get the input
        username = self.ui_login.lineEdit.text().strip()
        username = base64.urlsafe_b64encode(username.encode("UTF-8"))
        print(username)
        password = self.ui_login.lineEdit_2.text().strip()
        password = base64.urlsafe_b64encode(password.encode("UTF-8"))
        # account info
        USER_PWD = get_id_info()
        print(USER_PWD)
        if str(username) not in USER_PWD.keys():
            replay = QMessageBox.warning(self,"!", "Incorrect username", QMessageBox.Yes)
        else:
            # if success
            if USER_PWD.get(str(username)) == str(password):
                print("Jump to main window")
                shareInfo.loginWin = MainWindow_controller()
                shareInfo.loginWin.show()
                self.close()
            else:
                replay = QMessageBox.warning(self, "!", "Incorrect password", QMessageBox.Yes)
# signup interface
class win_Register(QMainWindow):
    def __init__(self, parent = None):
        super(win_Register, self).__init__(parent)
        self.ui_register = Ui_Dialog()
        self.ui_register.setupUi(self)
        self.init_slots()
        self.hidden_pwd()
    def hidden_pwd(self):
        self.ui_register.lineEdit_2.setEchoMode(QLineEdit.Password)
        self.ui_register.lineEdit_3.setEchoMode(QLineEdit.Password)
    # init
    def init_slots(self):
        self.ui_register.pushButton_2.clicked.connect(self.new_account)
        self.ui_register.pushButton_3.clicked.connect(self.cancel)

    # new_account
    def new_account(self):
        print("Create new account")
        USER_PWD = get_id_info()
        # print(USER_PWD)
        new_username = self.ui_register.lineEdit.text().strip()
        new_username = base64.urlsafe_b64encode(new_username.encode("UTF-8"))
        new_password = self.ui_register.lineEdit_2.text().strip()
        new_password = base64.urlsafe_b64encode(new_password.encode("UTF-8"))
        new_password_2 = self.ui_register.lineEdit_3.text().strip()
        new_password_2 = base64.urlsafe_b64encode(new_password_2.encode("UTF-8"))
        invicode = self.ui_register.lineEdit_4.text().strip()
        # Account already empty
        if new_username == "":
            replay = QMessageBox.warning(self, "!", "Username cannot be empty", QMessageBox.Yes)
        else:
            # Account already exists
            if new_username in USER_PWD.keys():
                replay = QMessageBox.warning(self, "!","Username already exists", QMessageBox.Yes)
            else:
                # pass
                if new_password == "":
                    replay = QMessageBox.warning(self, "!", "Password cannot be empty", QMessageBox.Yes)
                else:
                    if new_password_2 == "" or new_password_2 != new_password:
                        replay = QMessageBox.warning(self, "!", "Please confirm your password", QMessageBox.Yes)
                    else:
                        if invicode != "fyp2021wa03":# Successful
                            replay = QMessageBox.warning(self, "!", "Please confirm your Invitation Code", QMessageBox.Yes)
                        else:
                            print("Successful!")
                            sava_id_info(new_username, new_password)
                            replay = QMessageBox.warning(self, "!", "Successful！", QMessageBox.Yes)
                            # close
                            self.close()

    # cancellation
    def cancel(self):
        self.close()
# main functions// Calibration
class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.timer_camera_2 = QTimer()  # 初始化定时器
        self.timer_camera_1 = QTimer()
        self.timer_camera_3 = QTimer()
        self.timer_camera_Q = QTimer()
        self.timer_camera_Y = QTimer()
        self.cap = cv2.VideoCapture()  # 初始化摄像头
        self.CAM_NUM = shareInfo.CAM_NUM
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO
        #push button
        self.ui.pushButton.setText('RESET')
        self.clicked_counter = 0
        self.ui.pushButton.clicked.connect(self.buttonClicked_reset)
        self.ui.pushButton_2.setText('START')
        self.clicked_counter_2 = 0
        self.ui.pushButton_2.clicked.connect(self.slotCameraButton_2)  # monitoring mode
        self.ui.pushButton_3.setText('EXIT!')
        self.clicked_counter_3 = 0
        self.ui.pushButton_3.clicked.connect(self.buttonClicked_exit)
        self.ui.pushButton_4.setText('START')
        self.clicked_counter_4 = 0
        self.ui.pushButton_4.clicked.connect(self.buttonClicked_start)
        self.ui.pushButton_5.setText('ENTER')
        self.clicked_counter_5 = 0
        self.ui.pushButton_5.clicked.connect(self.buttonClicked_enter)
        self.ui.pushButton_7.setText('ENTER')
        self.clicked_counter_7 = 0
        self.ui.pushButton_7.clicked.connect(self.buttonClicked_offset)  # OFFSET
        self.ui.pushButton_6.setText('RESET')
        self.clicked_counter_6 = 0
        self.ui.pushButton_6.clicked.connect(self.buttonClicked_reset_s)
        self.ui.pushButton_9.setText('ENTER')
        self.clicked_counter_9 = 0
        self.ui.pushButton_9.clicked.connect(self.buttonClicked_QR)
        self.ui.pushButton_10.setText('START')
        self.clicked_counter_10 = 0
        self.ui.pushButton_10.clicked.connect(self.buttonClicked_yolo)  # YOLO5
        self.ui.pushButton_11.setText('RESET')
        self.clicked_counter_11 = 0
        self.ui.pushButton_11.clicked.connect(self.buttonClicked_reset_V)

        #radiobutton
        self.ui.radioButton.setText('Calibration Mode')
        self.ui.radioButton_2.setText('Monitor Mode')
        self.ui.radioButton.toggled.connect(self.onClicked_C)
        self.ui.radioButton_2.toggled.connect(self.onClicked_M)
        #self.ui.radioButton_3.toggled.connect(self.onClicked_S)
        #self.ui.radioButton_4.toggled.connect(self.onClicked_G)
        self.ui.radioButton_5.toggled.connect(self.onClicked_Q)
        self.ui.radioButton_6.toggled.connect(self.onClicked_Y)
        #timer
        self.timer_camera_2.timeout.connect(self.show_camera_2)
        self.timer_camera_1.timeout.connect(self.show_camera_1)
        self.timer_camera_Q.timeout.connect(self.show_camera_Q)
        self.timer_camera_Y.timeout.connect(self.show_camera_Y)
        #labal frame
        self.ui.label_7.setFrameShape(QtWidgets.QFrame.Box)
        self.ui.label_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ui.label_7.setFrameShape(QFrame.Box)
        self.ui.label_7.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 255, 255);background-color: rgb(255, 255, 240);')
        self.ui.label_9.setFrameShape(QtWidgets.QFrame.Box)
        self.ui.label_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ui.label_9.setFrameShape(QFrame.Box)
        self.ui.label_9.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 255, 255);background-color: rgb(255, 255, 240);')
        self.ui.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.ui.label_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ui.label_2.setFrameShape(QFrame.Box)
        self.ui.label_2.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 255, 255);background-color: rgb(255, 255, 240);')

        self.ui.comboBox.currentTextChanged.connect(self.markersize_change)
        self.ui.doubleSpinBox.valueChanged.connect(lambda x: self.change_val_video(x, 'brightBox'))
        self.ui.horizontalSlider.valueChanged.connect(lambda x: self.change_val_video(x, 'brightSlider'))
        self.ui.doubleSpinBox_2.valueChanged.connect(lambda x: self.change_val_video(x, 'colorBox'))
        self.ui.horizontalSlider_2.valueChanged.connect(lambda x: self.change_val_video(x, 'colorSlider'))
        self.ui.doubleSpinBox_3.valueChanged.connect(lambda x: self.change_val_video(x, 'contrastSpinBox'))
        self.ui.horizontalSlider_3.valueChanged.connect(lambda x: self.change_val_video(x, 'contrastSlider'))
        self.ui.doubleSpinBox_4.valueChanged.connect(lambda x: self.change_val_video(x, 'sharpBox'))
        self.ui.horizontalSlider_4.valueChanged.connect(lambda x: self.change_val_video(x, 'sharpSlider'))

        #self.Time_Enter = 0
        self.buttonClick = False
        self.buttonClick_2 = False
        self.buttonClicked_o = False
        self.buttonClicked_Q = False
        #self.grau_butten = False
        self.offsetx = shareInfo.offsetx
        self.offsety = shareInfo.offsety
        self.markersize = shareInfo.markersize
        #self.scaletime = shareInfo.scaletime
        self.load_video_settings()
        self.center()
    def load_video_settings(self):
        config_file = 'config/video_enhance.json'
        if not os.path.exists(config_file):
            brightness = 0.25
            color = 0.25,
            contrast = 0.25,
            sharpness = 0.25
            new_config = {"brightness": 0.25,
                          "color": 0.25,
                          "contrast": 0.25,
                          "sharpness": 0.25,
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            brightness = config['brightness']
            color = config['color']
            contrast = config['contrast']
            sharpness = config['sharpness']
            self.ui.doubleSpinBox.setValue(brightness)
            self.ui.doubleSpinBox_2.setValue(color)
            self.ui.doubleSpinBox_3.setValue(contrast)
            self.ui.doubleSpinBox_4.setValue(sharpness)

    def enhance(self,image):
        config_file = 'config/video_enhance.json'
        new_config = {"brightness": shareInfo.brightness,
                      "color": shareInfo.color,
                      "contrast": shareInfo.contrast,
                      "sharpness": shareInfo.sharpness
                      }
        new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(new_json)

        enh_bri = ImageEnhance.Brightness(image)
        brightness = shareInfo.brightness
        image_brightened = enh_bri.enhance(brightness)
        # image_brightened.show()

        # 色度增强
        enh_col = ImageEnhance.Color(image_brightened)
        color = shareInfo.color
        image_colored = enh_col.enhance(color)
        # image_colored.show()

        # 对比度增强
        enh_con = ImageEnhance.Contrast(image_colored)
        contrast = shareInfo.contrast
        image_contrasted = enh_con.enhance(contrast)
        # image_contrasted.show()

        # 锐度增强
        enh_sha = ImageEnhance.Sharpness(image_contrasted)
        sharpness = shareInfo.sharpness
        image_sharped = enh_sha.enhance(sharpness)
        # image_sharped.show()
        return image_sharped
    def markersize_change(self):
        self.markersize = float(self.ui.comboBox.currentText())
        shareInfo.markersize = float(self.ui.comboBox.currentText())
        self.ui.label_2.setText("markersize is "+str(self.markersize) +" cm")
    def click_clean(self):
        self.cap.release()
        self.timer_camera_1.stop()
        self.timer_camera_2.stop()
        self.timer_camera_3.stop()
        self.timer_camera_Q.stop()
        self.timer_camera_Y.stop()
        self.ui.label_7.clear()
        self.ui.label_9.clear()
        self.ui.label_5.clear()
        self.ui.label_6.clear()
        self.ui.textBrowser.clear()
        self.ui.pushButton_4.setText('START')
        self.ui.pushButton_2.setText('START')
        #self.ui.progressBar.setValue(0)
        QApplication.processEvents()
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    ###Funtion one !!!replay = QMessageBox.warning(self,"!", "Incorrect username or password", QMessageBox.Yes)
    def onClicked_C(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            self.click_clean()
            #self.grau_butten = False
            self.buttonClick = True
            self.buttonClick_2 = False
            msg = "Please Enter the offset between cam and machine arm\n" \
                  "X <---------(0,0) \n" \
                  "       -      |\n" \
                  "  +   (C)  -  |\n" \
                  "       +      v\n" \
                  "(480,640)     Y\n" \
                  "Please click START to start the camera"
            self.ui.label_2.setText(msg)
            #os.system("ml_scale_calculate.py")
            #print(pd.read_csv('markers_data2.csv', usecols=['x','y']).values)
            c = pd.read_csv('markers_data2.csv', usecols=['x','y']).values
            if c is []:
                replay = QMessageBox.warning(self, "!", "Please collect marker data first!!!", QMessageBox.Yes)
    def buttonClicked_start(self):
        if self.buttonClick == True:
            #self.Time_Enter = int(self.ui.lineEdit.text())
            # print(self.Time_Enter)
            ## Need to run at the same time
            #global maxtime
            #maxtime = self.Time_Enter
            # self.startThread()
            # self.show_camera_1()
            if self.timer_camera_1.isActive() == False:
                with open('markers_distance.csv', 'r+') as fp:
                    fp.truncate()
                    headers = ['index','ids',"x","y", 'distance_x', 'distance_y']
                    write = csv.writer(fp)
                    write.writerow(headers)
                    fp.close()

                #self.ui.progressBar.setMaximum(maxtime + 10)
                msg = "Please Enter the offset between cam and machine arm\n" \
                      "X <---------(0,0) \n" \
                      "       -      |\n" \
                      "  +   (C)  -  |\n" \
                      "       +      v\n" \
                      "(480,640)     Y\n" \
                      "calibrating... Please Wait"
                self.ui.label_2.setText(msg)
                self.openCamera()
            else:
                self.closeCamera_1()

        else:
            msgs = 'Choose a camera MODE first'
            self.ui.label_2.setText(msgs)
    def openCamera(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            msg = QMessageBox.Warning(self, u'Warning', u'ERROR Please Check!',
                                      buttons=QMessageBox.Ok,
                                      defaultButton=QMessageBox.Ok)
            self.ui.label_2.setText(msg)
        else:
            if self.buttonClick == True:
                #self.timebeg_1 = time.process_time()
                self.index = 0
                self.ui.pushButton_4.setText('STOP')
                msg = "Please Enter the offset between cam and machine arm\n" \
                      "X <---------(0,0) \n" \
                      "       -      |\n" \
                      "  +   (C)  -  |\n" \
                      "       +      v\n" \
                      "(480,640)     Y\n" \
                      "Please click STOP to stop the camera"
                self.ui.label_2.setText(msg)
                self.timer_camera_1.start(10)
            elif self.buttonClick_2 == True:
                self.ui.pushButton_2.setText('STOP')
                msg = "Please Enter the offset between cam and machine arm\n" \
                      "X <---------(0,0) \n" \
                      "       -      |\n" \
                      "  +   (C)  -  |\n" \
                      "       +      v\n" \
                      "(480,640)     Y\n" \
                      "Please click STOP to stop the camera"

                self.ui.label_2.setText(msg)
                self.timer_camera_2.start(10)


        # 关闭摄像头
    def show_camera_1(self):
        # load cam data
        cv_file = cv2.FileStorage("camera.yaml", cv2.FILE_STORAGE_READ)
        camera_matrix = cv_file.getNode("camera_matrix").mat()
        dist_matrix = cv_file.getNode("dist_coeff").mat()
        cv_file.release()
        dist = np.array(([[-0.01337232, 0.01314211, -0.00060755, -0.00497024, 0.08519319]]))
        newcameramtx = np.array([[484.55267334, 0., 325.60812827],
                                 [0., 480.50973511, 258.93040826],
                                 [0., 0., 1.]])
        mtx = np.array([[428.03839374, 0, 339.37509535],
                        [0., 427.81724311, 244.15085121],
                        [0., 0., 1.]])
        font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)
        CACHED_PTS = None
        CACHED_IDS = None
        Line_Pts = None

        Dist = []
        ret, frame = self.cap.read()
        h1 = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取视频帧的宽
        w1 = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h2 = int(h1 / 2)
        w2 = int(w1 / 2)
        # print(h1, w1, h2, w2)  # 480 640 240 320
        cam_coordinate = (int(h2), int(w2))
        # 纠正畸变
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_matrix, (h1, w1), 0, (h1, w1))
        frame = cv2.undistort(frame, camera_matrix, dist_matrix, None, newcameramtx)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian1 = cv2.GaussianBlur(gray, (5, 5), 0)
        gaussian1_enhance = Image.fromarray(np.uint8(gaussian1))
        gaussian1_enhance = self.enhance(gaussian1_enhance)

        #otsu
        #retval, gray_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # retval best threshold

        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        parameters = aruco.DetectorParameters_create()

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            np.asarray(gaussian1_enhance), aruco_dict, parameters=parameters)
        #corners, ids, rejectedImgPoints = aruco.detectMarkers(
        #    gaussian1, aruco_dict, parameters=parameters)
        if len(corners) <= 0:
            if CACHED_PTS is not None:
                corners = CACHED_PTS
        if len(corners) > 0:
            CACHED_PTS = corners
            if ids is not None:
                ids = ids.flatten()
                CACHED_IDS = ids
            else:
                if CACHED_IDS is not None:
                    ids = CACHED_IDS
            if len(corners) < 2:
                if len(CACHED_PTS) >= 2:
                    corners = CACHED_PTS
            for (markerCorner, markerId) in zip(corners, ids):
                # print("[INFO] Marker detected")
                corners_abcd = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners_abcd
                cX = ((topLeft[0] + bottomRight[0]) // 2)
                cY = ((topLeft[1] + bottomRight[1]) // 2)

                cv2.circle(frame, (int(cX), int(cY)), 4, (255, 0, 0), -1)
                cv2.circle(np.asarray(gaussian1_enhance), (int(cX), int(cY)), 4, (255, 0, 0), -1)
                Dist.append((int(cX), int(cY)))
                if len(Dist) == 0:
                    if Line_Pts is not None:
                        Dist = Line_Pts
                if len(Dist) == 2:
                    Line_Pts = Dist
                print(len(ids))
                if ids is not None and len(ids) ==1:
                    # 获取aruco返回的rvec旋转矩阵、tvec位移矩阵# markersize in cm
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners,shareInfo.markersize, camera_matrix, dist_matrix)
                    (rvec - tvec).any()  # get rid of that nasty numpy value array error
                    print("rvec",rvec)
                    print("tvec", tvec)
                    str_position = "MARKER Position x=%.6f  y=%.6f  z=%.6f" % (tvec[0][0][0], tvec[0][0][1], tvec[0][0][2])

                    #void solvePnP(InputArray objectPoints, InputArray imagePoints,
                    # InputArray cameraMatrix, InputArray distCoeffs,
                    # OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess=false,
                    # int flags = CV_ITERATIVE)


                    print(str_position)

                    #R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
                    #R_ct = R_ct.T

                    #R_flip = np.zeros((3,3), dtype = np.float32)
                    #R_flip[0, 0] = 1.0
                    #R_flip[1, 1] = -1.0
                    #R_flip[2, 2] = -1.0
                    #roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip*R_ct)
                    for i in range(rvec.shape[0]):
                        aruco.drawDetectedMarkers(frame, corners, ids)
                        aruco.drawAxis(frame, camera_matrix, dist_matrix, rvec[i, :, :], tvec[i, :, :], 1.5*shareInfo.markersize)

                        aruco.drawAxis(np.asarray(gaussian1_enhance), camera_matrix, dist_matrix, rvec[i, :, :], tvec[i, :, :], 1)
                        aruco.drawDetectedMarkers(np.asarray(gaussian1_enhance), corners, ids)


                        c = corners[i][0]
                        cx = float(c[:, 1].mean())
                        cy = float(c[:, 0].mean())
                        coordinate = (cx, cy)
                        cv2.circle(frame, (int(cy), int(cx)), 2, (255, 255, 0), 2)

                        # marker 中心与画面中心距离
                        p1 = np.array(cam_coordinate)
                        cv2.circle(frame, (int(p1[1]), int(p1[0])), 2, (255, 255, 0), 2)
                        p2 = np.array(coordinate)
                        p3 = p2 - p1
                        #print("distance")
                        cv2.line(frame,(int(p1[1]), int(p1[0])),(int(cy), int(cx)),(255, 255, 0),1 )
                        cv2.line(frame, (int(p1[1]), int(p1[0])), (int(cy), int(p1[0])), (255, 255, 0), 1)##x
                        cv2.line(frame, (int(cy), int(cx)), (int(cy), int(p1[0])), (255, 255, 0), 1)


                        #ml_scale_use.pix_scale(p2[0], p2[1], self.markersize)
                        #print(p1,p2,p3, self.markersize, ml_scale_use.x_scale,ml_scale_use.y_scale,shareInfo.x_scale,shareInfo.y_scale)

                        #distance_x = float(p3[0]) * (shareInfo.x_scale)
                        #distance_y = float(p3[1]) * (shareInfo.y_scale)
                        (topLeft, topRight, bottomRight, bottomLeft) = corners[0][0][0], corners[0][0][1], \
                                                                       corners[0][0][2], \
                                                                       corners[0][0][3]
                        topRightx = np.array((float(topRight[0]), float(topRight[1])))
                        bottomRightx = np.array((float(bottomRight[0]), float(bottomRight[1])))
                        bottomLeftx = np.array((float(bottomLeft[0]), float(bottomLeft[1])))
                        topLeftx = np.array((float(topLeft[0]), float(topLeft[1])))

                        right = topRightx - bottomRightx
                        left = topLeftx - bottomLeftx
                        top = topLeftx - topRightx
                        bottom = bottomLeftx - bottomRightx
                        rightdis = math.hypot(right[0], right[1])
                        leftdis = math.hypot(left[0], left[1])
                        topdis = math.hypot(top[0], top[1])
                        bottomdis = math.hypot(bottom[0], bottom[1])

                        average4 = (rightdis + leftdis + topdis + bottomdis) / 4
                        averagecol = (rightdis + leftdis) / 2
                        averagerow = (topdis + bottomdis) / 2

                        distance_x = shareInfo.markersize * float(p3[0]) / averagecol
                        distance_y = shareInfo.markersize * float(p3[1]) / averagerow
                        values = [self.index ,ids[0],float(p3[0]),float(p3[1]), float(distance_x), float(distance_y)]
                        print(p1,p2,p3, self.markersize,shareInfo.markersize/ averagecol,shareInfo.markersize/ averagerow,float(distance_x), float(distance_y))

                        self.ui.label_5.setText(str(float(distance_x)+ shareInfo.offsetx)+"cm")
                        self.ui.label_6.setText(str(float(distance_y) + shareInfo.offsety)+"cm")

                        with open('markers_distance.csv', 'a+', newline='') as fp:
                            write = csv.writer(fp)
                            write.writerow(values)
                            self.index = self.index +1
                        #self.printxydata()
                        cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        self.ui.textBrowser.setText("Id: " + str(ids) + str(str_position))

                else:
                    ##### DRAW "NO IDS" #####
                    cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    self.ui.textBrowser.setText("NO IDS")

        #if self.grau_butten == True:

        #else:
        show = cv2.resize(frame, (480, 360))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        cv2.line(show, (220, 180), (260, 180), (0, 0, 225), 1)
        cv2.line(show, (240, 160), (240, 200), (0, 0, 225), 1)
        show = cv2.rotate(show, cv2.ROTATE_90_CLOCKWISE)
        #cv2.imshow("im",show)
        if ids is not None:
            cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            self.ui.textBrowser.setText("Id: " + str(ids))
        else:
            ##### DRAW "NO IDS" #####
            cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            self.ui.textBrowser.setText("NO IDS")
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.ui.label_7.setPixmap(QPixmap.fromImage(showImage))

        show2 = cv2.resize(np.asarray(gaussian1_enhance), (480, 360))
        #show = cv2.resize(np.asarray(gaussian1), (640, 480))
        show2 = cv2.cvtColor(show2, cv2.COLOR_BGR2RGB)
        cv2.line(show2, (220, 180), (260, 180), (0, 0, 225), 1)
        cv2.line(show2, (240, 160), (240, 200), (0, 0, 225), 1)
        show2 = cv2.rotate(show2, cv2.ROTATE_90_CLOCKWISE)
        showImage = QImage(show2.data, show2.shape[1], show2.shape[0], QImage.Format_RGB888)
        self.ui.label_9.setPixmap(QPixmap.fromImage(showImage))

            #print(str(self.ui.progressBar.text()))
        if self.ui.pushButton_4.text() == "START":
            self.closeCamera_1()
            return

    def closeCamera_1(self):
        self.ui.pushButton_4.setText('START')
        self.cap.release()
        self.timer_camera_1.stop()
        self.ui.label_2.clear()
        #self.ui.label_7.clear()
        #self.ui.label_9.clear()
        self.ui.textBrowser.clear()
        self.datafilter()
        #self.printxydata()
        self.buttonClick_2 = False
    def datafilter(self):
        ilf = IsolationForest(n_estimators=100,
                              n_jobs=-1,
                              verbose=2,
                              )
        data = pd.read_csv('markers_distance.csv', index_col="index")
        data = data.fillna(0)
        X_cols = ["ids","x","y", "distance_x","distance_y"]
        print(data.shape)
        ilf.fit(data[X_cols])
        shape = data.shape[0]
        batch = 10 ** 6
        all_pred = []
        for i in range(int(shape / batch + 1)):
            start = i * batch
            end = (i + 1) * batch
            test = data[X_cols][start:end]
            # 预测
            pred = ilf.predict(test)
            all_pred.extend(pred)
        data['pred'] = all_pred
        with open('outliers.csv', 'a+', newline='') as fp:
            # 获取对象
            write = csv.writer(fp)
            headers = ["index", 'pred']
            write.writerow(headers)
        data.to_csv('outliers.csv', columns=["pred", ])
        df1 = pd.read_csv('markers_distance.csv')
        df2 = pd.read_csv('outliers.csv')
        all_csv = pd.merge(df1, df2, how='left', on=['index', 'index'])
        all_csv.dropna(axis=0, inplace=True)
        all_csv.drop_duplicates(inplace=True)
        # print(all_csv)
        # all_csv.to_csv("outliers2.csv", sep = ',', header = True,index = False)
        try:
            all_csv = all_csv.drop(all_csv[all_csv['pred'] == -1].index)
        finally:
            all_csv.to_csv("markers_distance.csv", sep=',', header=True, index=False)
        self.ml_scale_calculate()
        self.printxydata()
    ###print result
    def printxydata(self):
        msg = "Please Enter the offset between cam and machine arm\n" \
              "X <-----------(0,0) \n" \
              "        -       |\n" \
              "   +  (CAM)  -  |\n" \
              "        +       v\n" \
              "(480,640)       Y"
        self.ui.label_2.setText(msg)
        x = pd.read_csv('markers_distance.csv', usecols=['x']).values
        y = pd.read_csv('markers_distance.csv', usecols=['y']).values
        distance_x = pd.read_csv('markers_distance.csv', usecols=['distance_x']).values
        distance_y = pd.read_csv('markers_distance.csv', usecols=['distance_y']).values
        print('here', distance_x.mean(), distance_y.mean())
        ml_scale_use.pix_scale(x.mean(), y.mean(), self.markersize)
        #self.ui.label_5.setText(str(distance_x.mean() + self.offsetx) + "cm")
        self.ui.label_5.setText(str(ml_scale_use.x_prd_y[0][0]) + "cm")
        #self.ui.label_6.setText(str(distance_y.mean() + self.offsety) + "cm")
        self.ui.label_6.setText(str(ml_scale_use.y_prd_y[0][0]) + "cm")
        with open('markers_distance.csv', 'r+') as fp:
            fp.truncate()
            fp.close()
    ###Funtion TWO !!!
    def onClicked_M(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            self.click_clean()
            msg = "Please click START to start the camera"
            self.ui.label_2.setText(msg)
            self.buttonClick_2 = True
            self.buttonClick = False
    def slotCameraButton_2(self):
        if self.buttonClick_2 == True:
            if self.timer_camera_2.isActive() == False:
                # 打开摄像头并显示图像信息
                self.openCamera()
                # self.startThread()
            else:
                self.closeCamera_2()
    def show_camera_2(self):
        flag, self.image_2 = self.cap.read()
        gray = cv2.cvtColor(self.image_2, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(self.image_2, cv2.COLOR_BGR2GRAY)
        gaussian1 = cv2.GaussianBlur(gray, (5, 5), 0)
        gaussian1_enhance = Image.fromarray(np.uint8(gaussian1))
        gaussian1_enhance = self.enhance(gaussian1_enhance)


        #retval, self.image_2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #refresh
        QApplication.processEvents()

        show = cv2.resize(self.image_2, (480, 360))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        cv2.line(show, (230, 180), (250, 180), (0, 0, 225), 1)
        cv2.line(show, (240, 170), (240, 190), (0, 0, 225), 1)
        show = cv2.rotate(show, cv2.ROTATE_90_CLOCKWISE)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.ui.label_7.setPixmap(QPixmap.fromImage(showImage))

        show2 = cv2.resize(np.asarray(gaussian1_enhance), (480, 360))
        # show = cv2.resize(np.asarray(gaussian1), (640, 480))
        show2 = cv2.cvtColor(show2, cv2.COLOR_BGR2RGB)
        cv2.line(show2, (230, 180), (250, 180), (0, 0, 225), 1)
        cv2.line(show2, (240, 170), (240, 190), (0, 0, 225), 1)
        show2 = cv2.rotate(show2, cv2.ROTATE_90_CLOCKWISE)
        showImage = QImage(show2.data, show2.shape[1], show2.shape[0], QImage.Format_RGB888)
        self.ui.label_9.setPixmap(QPixmap.fromImage(showImage))


    def closeCamera_2(self):
        self.cap.release()
        self.timer_camera_2.stop()
        # self.ui.label_7.clear()
        msg = "Please click START to start the camera"
        self.ui.label_2.setText(msg)
        self.ui.pushButton_2.setText('START')
        self.ui.label_2.clear()
        self.ui.label_7.clear()
        self.ui.label_9.clear()
        # self.ui.lcdNumber.clear()


    ###setup
    ###EXUT
    def buttonClicked_exit(self):
        self.clicked_counter_3 += 1
        if self.clicked_counter_3 >= 1:
            sys.exit()
    ###choose cam
    def buttonClicked_enter(self):
        self.CAM_NUM = int(self.ui.lineEdit_2.text())
        shareInfo.CAM_NUM = self.CAM_NUM
    ###reset printed result
    def buttonClicked_reset(self):
        self.ui.label_5.setText("NULL")
        self.ui.label_6.setText("NULL")


    ###yolo5(new win)
    def onClicked_Y(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            self.click_clean()
            msg = "Recognition Mode Using YOLOv5"
            self.ui.label_2.setText(msg)
            time.sleep(1)
            self.buttonClicked_Y = True
    def openCamera_Y(self):
        flag = self.cap.open(self.CAM_NUM,cv2.CAP_DSHOW)
        if flag == False:
            msg = QMessageBox.Warning(self, u'Warning', u'ERROR Please Check!',
                                          buttons=QMessageBox.Ok,
                                          defaultButton=QMessageBox.Ok)
            self.ui.label_2.setText(msg)
        else:
            self.timer_camera_Y.start(10)
        #open cam to scan QR-code//     qr-num
    def buttonClicked_yolo(self):
        if self.buttonClicked_Y == True:
            print("jump to yolo")
            #shareInfo.createWin = win_Register()
            #shareInfo.createWin.show()
            self.close()
            shareInfo.loginWin = MainWindow()
            shareInfo.loginWin.show()

            """
            if self.timer_camera_Y.isAccreateWintive() == False:
                # 打开摄像头并显示图像信息
                self.ui.pushButton_10.setText('STOP')
                self.openCamera_Y()
                 # self.startThread()
            else:
                self.closeCamera_Y()
            """
    def show_camera_Y(self):
        flag, self.image_y = self.cap.read()
        QApplication.processEvents()

        show_y = cv2.resize(self.image_y, (640, 480))
        show_y = cv2.cvtColor(show_y, cv2.COLOR_BGR2RGB)
        cv2.line(show_y, (315, 240), (325, 240), (0, 0, 225), 1)
        cv2.line(show_y, (320, 235), (320, 245), (0, 0, 225), 1)
        show_y = cv2.rotate(show_y, cv2.ROTATE_90_CLOCKWISE)
        showImage_y = QImage(show_y.data, show_y.shape[1], show_y.shape[0], QImage.Format_RGB888)
        self.ui.label_7.setPixmap(QPixmap.fromImage(showImage_y))
    def closeCamera_Y(self):
        self.cap.release()
        self.timer_camera_1.stop()
        self.timer_camera_2.stop()
        self.timer_camera_3.stop()
        self.timer_camera_Q.stop()
        self.timer_camera_Y.stop()
        msg = "Please click START to start the camera"
        self.ui.label_2.setText(msg)
        self.ui.pushButton_10.setText('START')
        self.ui.label_7.clear()
        self.ui.label_9.clear()

    ###offset settings
    def buttonClicked_offset(self):

        self.offsetx = float(self.ui.lineEdit_3.text())
        self.offsety = float(self.ui.lineEdit_4.text())
        shareInfo.offsetx = self.offsetx
        shareInfo.offsety = self.offsety
        self.ui.label_2.setText("Entered: -->\n"\
                                "offsetx= "+str(self.offsetx)+"\n"\
                                "offsety= "+str(self.offsety)+"\n")
        self.buttonClicked_o = True
    def buttonClicked_reset_s(self):
        self.ui.lineEdit_3.setText(str(self.offsetx))
        self.ui.lineEdit_4.setText(str(self.offsety))
        self.offsetx = shareInfo.offsetx
        self.offsety = shareInfo.offsety
        self.ui.label_2.setText("Reset!: -->\n" \
                                    "offsetx= " + str(self.offsetx) + "\n" \
                                                                      "offsety= " + str(self.offsety) + "\n")

    #QR-CODE
    def onClicked_Q(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            self.click_clean()
            msg = "Scan QR - CODE to set up the parameters"
            self.ui.label_2.setText(msg)
            self.buttonClicked_Q = True
            if self.timer_camera_Q.isActive() == False:
                #print("here")
                self.openCamera_Q()
                # self.startThread()
            else:
                self.closeCamera_Q()
    def openCamera_Q(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            msg = QMessageBox.Warning(self, u'Warning', u'ERROR Please Check!',
                                          buttons=QMessageBox.Ok,
                                          defaultButton=QMessageBox.Ok)
            self.ui.label_2.setText(msg)
        else:
            self.timer_camera_Q.start(10)
        #open cam to scan QR-code//     qr-num
    def show_camera_Q(self):
        flag, self.image_q = self.cap.read()
        QApplication.processEvents()


        barcodes = pyzbar.decode(self.image_q)
        self.barcodeData = ''
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(self.image_q, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.barcodeData = str(barcode.data.decode("utf-8"))
            #print(self.barcodeData)

        if self.barcodeData != '' :
            barcodeType = barcode.type
            self.barcodeData_list = self.barcodeData.split(",")
            if self.barcodeData_list[0] == "O":
                self.mode = 0
                msg = "Calibration Mode - Original image"
                self.ui.label_2.setText(msg)
                time = self.barcodeData_list[1]
                self.ui.lineEdit.setText(time)
            elif self.barcodeData_list[0] == "G":
                self.mode = 1
                msg = "Calibration Mode - Gray-Otsu image"
                self.ui.label_2.setText(msg)
                time = self.barcodeData_list[1]
                self.ui.lineEdit.setText(time)
            elif self.barcodeData_list[0] == "S":
                self.mode = 2
                msg = "Scale Setting Mode"
                self.ui.label_2.setText(msg)
                offsetx = self.barcodeData_list[1]
                self.ui.lineEdit_3.setText(offsetx)
                offsety = self.barcodeData_list[2]
                self.ui.lineEdit_4.setText(offsety)
                size = self.barcodeData_list[3]
                self.ui.lineEdit_3.setText(size)
                time = self.barcodeData_list[4]
                self.ui.lineEdit_5.setText(time)
            elif self.barcodeData_list[0] == "M":
                self.mode = 3
                msg = "Monitor Mode"
                self.ui.label_2.setText(msg)

            with open('qrcode.csv', 'a+') as fp:
                headers = [self.barcodeData, barcodeType, datetime.datetime.now()]
                write = csv.writer(fp)
                write.writerow(headers)
                fp.close()
            text = "{} ({})".format(self.barcodeData, barcodeType)
            cv2.putText(self.image_q, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            self.ui.textBrowser.setText(text)

        show_q = cv2.resize(self.image_q, (640, 480))
        show_q = cv2.cvtColor(show_q, cv2.COLOR_BGR2RGB)
        cv2.line(show_q, (315, 240), (325, 240), (0, 0, 225), 1)
        cv2.line(show_q, (320, 235), (320, 245), (0, 0, 225), 1)
        show_q = cv2.rotate(show_q, cv2.ROTATE_90_CLOCKWISE)
        showImage_q = QImage(show_q.data, show_q.shape[1], show_q.shape[0], QImage.Format_RGB888)
        self.ui.label_7.setPixmap(QPixmap.fromImage(showImage_q))
    def closeCamera_Q(self):
        self.cap.release()
        self.timer_camera_Q.stop()
        self.ui.label_2.clear()
        self.ui.label_7.clear()
        self.ui.label_9.clear()
    def buttonClicked_QR(self):
        if self.buttonClicked_Q == True:
            self.closeCamera_Q()
            if self.mode == 0: ##O
                #self.grau_butten = False
                self.buttonClick = True
                self.buttonClick_2 = False
                #os.system("ml_scale_calculate.py")
                #self.ml_scale_calculate()
                self.buttonClicked_start()
            elif self.mode == 1: ##G
                #self.grau_butten = True
                self.buttonClick = True
                self.buttonClick_2 = False
                #os.system("ml_scale_calculate.py")
                #self.ml_scale_calculate()
                self.buttonClicked_start()
            elif self.mode == 2: ##S
                self.buttonClicked_o = True
                self.buttonClicked_scale()
            elif self.mode == 3: ##M

                self.buttonClick_2 = True
                self.buttonClick = False
                self.slotCameraButton_2()

        else:
            msg = "Choose QR Mode First!"
            self.ui.label_2.setText(msg)
    #scale px -> cm/mm
    def ml_scale_calculate(self):

        x = pd.read_csv('markers_distance.csv', usecols=['x']).values
        y = pd.read_csv('markers_distance.csv', usecols=['y']).values
        d_x = pd.read_csv('markers_distance.csv', usecols=['distance_x']).values
        d_y = pd.read_csv('markers_distance.csv', usecols=['distance_y']).values
        ###ransac
        # Fit line using all data
        lr = linear_model.LinearRegression()
        lr.fit(x, d_x)

        # Robustly fit linear model with RANSAC algorithm
        ransac = linear_model.RANSACRegressor()
        ransac.fit(x, d_x)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        # Predict data of estimated models
        line_X = np.arange(x.min(), x.max())[:, np.newaxis]
        line_y = lr.predict(line_X)
        line_y_ransac = ransac.predict(line_X)

        # Compare estimated coefficients
        print("Estimated coefficients (true, linear regression, RANSAC):")
        print(lr.coef_, ransac.estimator_.coef_)

        lw = 2
        plt.figure()
        plt.scatter(
            x[inlier_mask], d_x[inlier_mask], color="yellowgreen", marker=".", label="Inliers-x"
        )
        plt.scatter(
            x[outlier_mask], d_x[outlier_mask], color="gold", marker=".", label="Outliers-x"
        )
        plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor-x")
        plt.plot(
            line_X,
            line_y_ransac,
            color="cornflowerblue",
            linewidth=lw,
            label="RANSAC regressor-x",
        )
        plt.legend(loc='lower left')
        plotPath = r'.\datax.png'
        plt.savefig(plotPath)
        jpg = QtGui.QPixmap(plotPath).scaled(self.ui.label_7.width(), self.ui.label_7.height())
        self.ui.label_7.setPixmap(jpg)
        # 存储模型
        with open('./linear2_x.pkl', 'wb') as f:
            pickle.dump(ransac, f)
            print("save success！")
        ###ransac for x

        ###ransac
        # Fit line using all data
        lr = linear_model.LinearRegression()
        lr.fit(y, d_y)
        # Robustly fit linear model with RANSAC algorithm
        ransac = linear_model.RANSACRegressor()
        ransac.fit(y, d_y)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        # Predict data of estimated models
        line_X = np.arange(y.min(), y.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)
        line_y = lr.predict(line_X)

        # Compare estimated coefficients
        print("Estimated coefficients (true, linear regression, RANSAC):")
        print(lr.coef_, ransac.estimator_.coef_)

        lw = 2
        plt.figure()
        plt.scatter(
            y[inlier_mask], d_y[inlier_mask], color="red", marker=".", label="Inliers-y"
        )
        plt.scatter(
            y[outlier_mask], d_y[outlier_mask], color="blue", marker=".", label="Outliers-y"
        )
        plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor-y")
        plt.plot(
            line_X,
            line_y_ransac,
            color="cornflowerblue",
            linewidth=lw,
            label="RANSAC regressor-y",
        )
        # 存储模型
        with open('./linear2_y.pkl', 'wb') as f:
            pickle.dump(ransac, f)
            print("save success！")

        plt.legend(loc='lower left')
        plotPath = r'.\datay.png'
        plt.savefig(plotPath)
        jpg = QtGui.QPixmap(plotPath).scaled(self.ui.label_9.width(), self.ui.label_9.height())
        self.ui.label_9.setPixmap(jpg)

    def buttonClicked_reset_V(self):
        shareInfo.brightness =0.5
        shareInfo.color = 2
        shareInfo.contrast = 2
        shareInfo.sharpness = 2
        self.ui.doubleSpinBox.setValue(shareInfo.brightness)
        self.ui.doubleSpinBox_2.setValue(shareInfo.color)
        self.ui.doubleSpinBox_3.setValue(shareInfo.contrast)
        self.ui.doubleSpinBox_4.setValue(shareInfo.sharpness)

    def change_val_video(self, x, flag):
        if flag == 'brightBox':
            self.ui.horizontalSlider.setValue(int(x*10))
            print(x)
        elif flag == 'brightSlider':
            self.ui.doubleSpinBox.setValue(x/10)
            shareInfo.brightness= x/10
            self.ui.label_2.setText("brightness :  "+str(shareInfo.brightness)+"\n"+"color :  "+str(shareInfo.color)+"\n"+"contrast :  "+str(shareInfo.contrast)+"\n"+"sharpness :  "+str(shareInfo.sharpness))
            print(x,"spin")
        elif flag == 'colorBox':
            self.ui.horizontalSlider_2.setValue(int(x*10))
            print(x)
        elif flag == 'colorSlider':
            self.ui.doubleSpinBox_2.setValue(x/10)
            shareInfo.color = x/10
            self.ui.label_2.setText("brightness :  " + str(shareInfo.brightness) +"\n"+"color :  " + str(
                shareInfo.color) +"\n"+ "contrast :  " + str(shareInfo.contrast) +"\n"+ "sharpness :  " + str(
                shareInfo.sharpness))
            print(x)
        elif flag == 'contrastSpinBox':
            self.ui.horizontalSlider_3.setValue(int(x*10))
            print(x)
        elif flag == 'contrastSlider':
            self.ui.doubleSpinBox_3.setValue(x/10)
            shareInfo.contrast = x / 10
            self.ui.label_2.setText("brightness :  " + str(shareInfo.brightness) +"\n"+ "color :  " + str(
                shareInfo.color) +"\n"+ "contrast :  " + str(shareInfo.contrast) +"\n"+ "sharpness :  " + str(
                shareInfo.sharpness))
            print(x)
        elif flag == 'sharpBox':
            self.ui.horizontalSlider_4.setValue(int(x*10))
            print(x)
        elif flag == 'sharpSlider':
            self.ui.doubleSpinBox_4.setValue(x/10)
            shareInfo.sharpness = x / 10
            self.ui.label_2.setText("brightness :  " + str(shareInfo.brightness) + "\n" + "color :  " + str(
                shareInfo.color) + "\n" + "contrast :  " + str(shareInfo.contrast) + "\n" + "sharpness :  " + str(
                shareInfo.sharpness))
            print(x)
        else:
            pass
#yolo thread
class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # emit signals
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov5n.pt'           # weight
        self.current_weight = './yolov5n.pt'    # weight
        self.source = '1'                       # cam
        self.conf_thres = 0.25                  # conf
        self.iou_thres = 0.45                   # iou
        self.jump_out = False                   # stop
        self.is_continue = True                 # pause
        self.percent_length = 1000              # progress bar
        self.rate_check = True                  # dalay OPEN/NO
        self.rate = 100                         # dalay HZ

    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

        # Initialize
        try:
            print("test")
            device = select_device(device)
            half &= device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if half:
                model.half()  # to FP16

            # Dataloader
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            count = 0

            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)
            while True:
                # 手动停止
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('STOP!')
                    break
                # change model
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                    if half:
                        model.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    self.current_weight = self.weights
                # stop
                if self.is_continue:
                    path, img, im0s, self.vid_cap = next(dataset)
                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
                    count += 1

                    if count % 30 == 0 and count >= 30:
                        fps = int(30/(time.time()-start_time))
                        self.send_fps.emit('fps：'+str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    statistic_dic = {name: 0 for name in names}
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img, augment=augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                statistic_dic[names[c]] += 1
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                # im0 = plot_one_box_PIL(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)  # 中文标签画框，但是耗时会增加
                                plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                             line_thickness=line_thickness)


                    if self.rate_check:
                        time.sleep(1/self.rate)
                    # print(type(im0s))
                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_statistic.emit(statistic_dic)
                    if percent == self.percent_length:
                        self.send_percent.emit(0)
                        self.send_msg.emit('Finish')

                        break

        except Exception as e:
            self.send_msg.emit('%s' % e)
#yolo interface
class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False
        self.setWindowFlags(Qt.CustomizeWindowHint)
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        self.closeButton.clicked.connect(self.close)


        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # model setup
        self.ComboBox_weight.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.ComboBox_weight.clear()
        self.ComboBox_weight.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5
        self.det_thread = DetThread()
        self.model_type = self.ComboBox_weight.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.det_thread.source = '1'
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.rtspButton.clicked.connect(self.chose_rtsp)
        self.pushButton.clicked.connect(self.reset)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)
        self.confSlider.setValue(25)
        self.confSpinBox.setValue(0.25)
        self.iouSlider.setValue(45)
        self.iouSpinBox.setValue(0.45)

        self.ComboBox_weight.currentTextChanged.connect(self.change_model)
        # self.ComboBox_weight.currentTextChanged.connect(lambda x: self.statistic_msg('model %s' % x))
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.load_setting()
    def reset(self):
        self.search_pt()
        self.stop()
        iou = 0.25
        conf = 0.45
        rate = 10
        check = 0
        self.det_thread.source = '1'
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check
        MessageBox(
            self.closeButton, title='WARNING！', text='ALL PARAMETERS WILL BE RESET', time=2000, auto=True).exec_()
        self.statistic_msg('RESET!!!')
    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.ComboBox_weight.clear()
            self.ComboBox_weight.addItems(self.pt_list)
    def checkrate(self):
        if self.checkBox.isChecked():
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False
    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='！', text='Please wait', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)
    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='！', text='Please wait', time=2000, auto=True).exec_()
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('Loading Cam：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)
    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.25
            conf = 0.45
            rate = 10
            check = 0
            new_config = {"iou": 0.25,
                          "conf": 0.45,
                          "rate": 10,
                          "check": 0
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            iou = config['iou']
            conf = config['conf']
            rate = config['rate']
            check = config['check']
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check
    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
            print(x)
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
            print(x)
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
            print(x)
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
            self.det_thread.iou_thres = x/100
            print(x)
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
            print(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
            print(x)
        else:
            pass
    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        self.qtimer.start(3000)
    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
    def change_model(self, x):
        self.model_type = self.ComboBox_weight.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Model change to %s' % x)
    def open_file(self):

        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Choose Pic or Video', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('Loading file：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)

            self.stop()
    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()
    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'CAM' if source.isnumeric() else source
            self.statistic_msg('Detecting >> Model：{}，Input File：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('PAUSE')
    def stop(self):
        self.det_thread.jump_out = True
        self.out_video.clear()
        self.raw_video.clear()
    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True
    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)
            # QMouseEvent.accept()
    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        # self.setCursor(QCursor(Qt.ArrowCursor))
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()

            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            if results != []:
                self.resultWidget.setStyleSheet("background-color:red")
                self.resultWidget.addItems(results)
            else:
                self.resultWidget.setStyleSheet("background-color: rgba(12, 28, 77, 0)")
                self.resultWidget.addItems(results)
        except Exception as e:
            print(repr(e))
    def closeEvent(self, event):

        self.det_thread.jump_out = True
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        shareInfo.loginWin = MainWindow_controller()
        shareInfo.loginWin.show()
        self.close()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    shareInfo.loginWin = win_Login()
    #shareInfo.loginWin = MainWindow_controller()
    shareInfo.loginWin.show()
    sys.exit(app.exec_())

