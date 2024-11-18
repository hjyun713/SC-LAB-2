import csv
import os
import sys
import threading
import time
import socket
from datetime import datetime
from collections import Counter

import cv2
import numpy as np
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from pyqtgraph import PlotWidget, mkPen
from serial import Serial
import torch
import torch.nn as nn

mode = 1
test_mode = True


class FormWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(805, 519)
        MainWindow.setMinimumSize(QSize(805, 519))
        self.centralwidget = QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_9 = QGridLayout(self.centralwidget)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.splitter_16 = QSplitter(parent=self.centralwidget)
        self.splitter_16.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_16.setObjectName("splitter_16")
        self.splitter_13 = QSplitter(parent=self.splitter_16)
        self.splitter_13.setMinimumSize(QSize(521, 501))
        self.splitter_13.setOrientation(Qt.Orientation.Vertical)
        self.splitter_13.setObjectName("splitter_13")
        self.splitter_12 = QSplitter(parent=self.splitter_13)
        self.splitter_12.setMinimumSize(QSize(521, 401))
        self.splitter_12.setOrientation(Qt.Orientation.Vertical)
        self.splitter_12.setObjectName("splitter_12")
        self.PLOT = PlotWidget(parent=self.splitter_12)
        self.PLOT.setMinimumSize(QSize(521, 371))
        self.PLOT.setFrameShape(QFrame.Shape.Box)
        self.PLOT.setFrameShadow(QFrame.Shadow.Plain)
        self.PLOT.setObjectName("PLOT")
        self.AI = QLabel(parent=self.splitter_12)
        self.AI.setMinimumSize(QSize(521, 21))
        self.AI.setMaximumSize(QSize(16777215, 21))
        self.AI.setFrameShape(QFrame.Shape.Box)
        self.AI.setText("")
        self.AI.setObjectName("AI")
        self.splitter_9 = QSplitter(parent=self.splitter_13)
        self.splitter_9.setMinimumSize(QSize(521, 90))
        self.splitter_9.setMaximumSize(QSize(16777215, 90))
        self.splitter_9.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_9.setObjectName("splitter_9")
        self.AX = QGroupBox(parent=self.splitter_9)
        self.AX.setMinimumSize(QSize(204, 90))
        self.AX.setMaximumSize(QSize(16777215, 90))
        self.AX.setStyleSheet("font: 9pt \"Arial\";")
        self.AX.setObjectName("AX")
        self.gridLayout_3 = QGridLayout(self.AX)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.splitter_7 = QSplitter(parent=self.AX)
        self.splitter_7.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_7.setObjectName("splitter_7")
        self.label_3 = QLabel(parent=self.splitter_7)
        self.label_3.setMinimumSize(QSize(40, 22))
        self.label_3.setMaximumSize(QSize(40, 22))
        self.label_3.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.slider_x = QSlider(parent=self.splitter_7)
        self.slider_x.setMinimum(50)
        self.slider_x.setMaximum(200)
        self.slider_x.setProperty("value", 50)
        self.slider_x.setOrientation(Qt.Orientation.Horizontal)
        self.slider_x.setObjectName("slider_x")
        self.spin_x = QSpinBox(parent=self.splitter_7)
        self.spin_x.setMinimumSize(QSize(50, 22))
        self.spin_x.setMaximumSize(QSize(50, 22))
        self.spin_x.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin_x.setMinimum(50)
        self.spin_x.setMaximum(200)
        self.spin_x.setDisplayIntegerBase(10)
        self.spin_x.setObjectName("spin_x")
        self.gridLayout_3.addWidget(self.splitter_7, 0, 0, 1, 1)
        self.splitter_8 = QSplitter(parent=self.AX)
        self.splitter_8.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_8.setObjectName("splitter_8")
        self.label_4 = QLabel(parent=self.splitter_8)
        self.label_4.setMinimumSize(QSize(40, 22))
        self.label_4.setMaximumSize(QSize(40, 22))
        self.label_4.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.slider_y = QSlider(parent=self.splitter_8)
        self.slider_y.setMinimum(10)
        self.slider_y.setMaximum(8000)
        self.slider_y.setProperty("value", 200)
        self.slider_y.setOrientation(Qt.Orientation.Horizontal)
        self.slider_y.setObjectName("slider_y")
        self.spin_y = QSpinBox(parent=self.splitter_8)
        self.spin_y.setMinimumSize(QSize(50, 22))
        self.spin_y.setMaximumSize(QSize(50, 22))
        self.spin_y.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin_y.setMinimum(10)
        self.spin_y.setMaximum(8000)
        self.spin_y.setProperty("value", 200)
        self.spin_y.setObjectName("spin_y")
        self.gridLayout_3.addWidget(self.splitter_8, 1, 0, 1, 1)
        self.GS = QGroupBox(parent=self.splitter_9)
        self.GS.setMinimumSize(QSize(141, 90))
        self.GS.setMaximumSize(QSize(141, 90))
        self.GS.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.GS.setStyleSheet("font: 9pt \"Arial\";")
        self.GS.setAlignment(Qt.AlignmentFlag.AlignLeading | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.GS.setFlat(False)
        self.GS.setObjectName("GS")
        self.gridLayout_4 = QGridLayout(self.GS)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.splitter_10 = QSplitter(parent=self.GS)
        self.splitter_10.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_10.setObjectName("splitter_10")
        self.CH = QComboBox(parent=self.splitter_10)
        self.CH.setMinimumSize(QSize(71, 22))
        self.CH.setMaximumSize(QSize(71, 22))
        self.CH.setObjectName("CH")
        self.CH.addItem("")
        self.COLOR = QPushButton(parent=self.splitter_10)
        self.COLOR.setMinimumSize(QSize(41, 22))
        self.COLOR.setMaximumSize(QSize(41, 22))
        self.COLOR.setStyleSheet("font: bold 10pt \"Consolas\";\n"
                                 "color: rgb(255, 0, 0);")
        self.COLOR.setObjectName("COLOR")
        self.gridLayout_4.addWidget(self.splitter_10, 0, 0, 1, 1)
        self.splitter_11 = QSplitter(parent=self.GS)
        self.splitter_11.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_11.setObjectName("splitter_11")
        self.label_5 = QLabel(parent=self.splitter_11)
        self.label_5.setMinimumSize(QSize(71, 22))
        self.label_5.setMaximumSize(QSize(71, 22))
        self.label_5.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.LINEWIDTH = QDoubleSpinBox(parent=self.splitter_11)
        self.LINEWIDTH.setMinimumSize(QSize(41, 22))
        self.LINEWIDTH.setMaximumSize(QSize(41, 22))
        self.LINEWIDTH.setDecimals(1)
        self.LINEWIDTH.setMinimum(0.5)
        self.LINEWIDTH.setMaximum(10.0)
        self.LINEWIDTH.setProperty("value", 1.0)
        self.LINEWIDTH.setObjectName("LINEWIDTH")
        self.gridLayout_4.addWidget(self.splitter_11, 1, 0, 1, 1)
        self.FILTER = QGroupBox(parent=self.splitter_9)
        self.FILTER.setMinimumSize(QSize(161, 90))
        self.FILTER.setMaximumSize(QSize(161, 90))
        self.FILTER.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.FILTER.setStyleSheet("font: 9pt \"Arial\";")
        self.FILTER.setAlignment(
            Qt.AlignmentFlag.AlignLeading | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.FILTER.setFlat(False)
        self.FILTER.setObjectName("FILTER")
        self.gridLayout_6 = QGridLayout(self.FILTER)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.F1 = QRadioButton(parent=self.FILTER)
        self.F1.setChecked(True)
        self.F1.setObjectName("F1")
        self.gridLayout_5.addWidget(self.F1, 0, 0, 1, 1)
        self.F2 = QRadioButton(parent=self.FILTER)
        self.F2.setObjectName("F2")
        self.gridLayout_5.addWidget(self.F2, 0, 1, 1, 1)
        self.F3 = QRadioButton(parent=self.FILTER)
        self.F3.setObjectName("F3")
        self.gridLayout_5.addWidget(self.F3, 1, 0, 1, 1)
        self.F4 = QRadioButton(parent=self.FILTER)
        self.F4.setObjectName("F4")
        self.gridLayout_5.addWidget(self.F4, 1, 1, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)
        self.splitter_15 = QSplitter(parent=self.splitter_16)
        self.splitter_15.setMinimumSize(QSize(261, 501))
        self.splitter_15.setMaximumSize(QSize(261, 16777215))
        self.splitter_15.setOrientation(Qt.Orientation.Vertical)
        self.splitter_15.setObjectName("splitter_15")
        self.tabWidget = QTabWidget(parent=self.splitter_15)
        self.tabWidget.setMinimumSize(QSize(261, 101))
        self.tabWidget.setMaximumSize(QSize(261, 100))
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        self.tabWidget.setFont(font)
        self.tabWidget.setAutoFillBackground(True)
        self.tabWidget.setStyleSheet("font: 9pt \"Arial\";")
        self.tabWidget.setObjectName("tabWidget")
        self.tab_1 = QWidget()
        self.tab_1.setObjectName("tab_1")
        self.gridLayout_2 = QGridLayout(self.tab_1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QLabel(parent=self.tab_1)
        self.label.setMinimumSize(QSize(116, 24))
        self.label.setMaximumSize(QSize(116, 24))
        self.label.setStyleSheet("font: bold 9pt \"Arial\";")
        self.label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.PORT_TCP = QLineEdit(parent=self.tab_1)
        self.PORT_TCP.setMinimumSize(QSize(116, 24))
        self.PORT_TCP.setMaximumSize(QSize(116, 24))
        self.PORT_TCP.setStyleSheet("font: 9pt \"Arial\";")
        self.PORT_TCP.setInputMask("")
        self.PORT_TCP.setText("50002")
        self.PORT_TCP.setMaxLength(32767)
        self.PORT_TCP.setEchoMode(QLineEdit.EchoMode.Normal)
        self.PORT_TCP.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.PORT_TCP.setDragEnabled(True)
        self.PORT_TCP.setPlaceholderText("")
        self.PORT_TCP.setObjectName("PORT_TCP")
        self.gridLayout_2.addWidget(self.PORT_TCP, 0, 1, 1, 1)
        self.OPEN_TCP = QPushButton(parent=self.tab_1)
        self.OPEN_TCP.setMinimumSize(QSize(116, 26))
        self.OPEN_TCP.setMaximumSize(QSize(116, 26))
        self.OPEN_TCP.setObjectName("OPEN_TCP")
        self.gridLayout_2.addWidget(self.OPEN_TCP, 1, 0, 1, 1)
        self.CLOSE_TCP = QPushButton(parent=self.tab_1)
        self.CLOSE_TCP.setMinimumSize(QSize(116, 26))
        self.CLOSE_TCP.setMaximumSize(QSize(116, 26))
        self.CLOSE_TCP.setObjectName("CLOSE_TCP")
        self.gridLayout_2.addWidget(self.CLOSE_TCP, 1, 1, 1, 1)
        self.tabWidget.addTab(self.tab_1, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout = QGridLayout(self.tab_2)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QLabel(parent=self.tab_2)
        self.label_2.setMinimumSize(QSize(116, 24))
        self.label_2.setMaximumSize(QSize(116, 24))
        self.label_2.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.label_2.setStyleSheet("font: bold 9pt \"Arial\";")
        self.label_2.setTextFormat(Qt.TextFormat.AutoText)
        self.label_2.setScaledContents(False)
        self.label_2.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.PORT_EMG = QLineEdit(parent=self.tab_2)
        self.PORT_EMG.setMinimumSize(QSize(116, 24))
        self.PORT_EMG.setMaximumSize(QSize(116, 24))
        self.PORT_EMG.setInputMask("")
        self.PORT_EMG.setText("3")
        self.PORT_EMG.setMaxLength(32767)
        self.PORT_EMG.setEchoMode(QLineEdit.EchoMode.Normal)
        self.PORT_EMG.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.PORT_EMG.setDragEnabled(True)
        self.PORT_EMG.setPlaceholderText("")
        self.PORT_EMG.setObjectName("PORT_EMG")
        self.gridLayout.addWidget(self.PORT_EMG, 0, 1, 1, 1)
        self.OPEN_EMG = QPushButton(parent=self.tab_2)
        self.OPEN_EMG.setMinimumSize(QSize(116, 26))
        self.OPEN_EMG.setMaximumSize(QSize(116, 26))
        self.OPEN_EMG.setAutoFillBackground(False)
        self.OPEN_EMG.setAutoRepeat(False)
        self.OPEN_EMG.setDefault(False)
        self.OPEN_EMG.setFlat(False)
        self.OPEN_EMG.setObjectName("OPEN_EMG")
        self.gridLayout.addWidget(self.OPEN_EMG, 1, 0, 1, 1)
        self.CLOSE_EMG = QPushButton(parent=self.tab_2)
        self.CLOSE_EMG.setMinimumSize(QSize(116, 26))
        self.CLOSE_EMG.setMaximumSize(QSize(116, 26))
        self.CLOSE_EMG.setObjectName("CLOSE_EMG")
        self.gridLayout.addWidget(self.CLOSE_EMG, 1, 1, 1, 1)
        self.tabWidget.addTab(self.tab_2, "EMG Sensor")
        self.DL = QGroupBox(parent=self.splitter_15)
        self.DL.setMinimumSize(QSize(261, 61))
        self.DL.setMaximumSize(QSize(261, 61))
        self.DL.setStyleSheet("font: 9pt \"Arial\";")
        self.DL.setObjectName("DL")
        self.gridLayout_7 = QGridLayout(self.DL)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.splitter_14 = QSplitter(parent=self.DL)
        self.splitter_14.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_14.setObjectName("splitter_14")
        self.LSTART = QPushButton(parent=self.splitter_14)
        self.LSTART.setMinimumSize(QSize(116, 26))
        self.LSTART.setMaximumSize(QSize(116, 26))
        self.LSTART.setCheckable(True)
        self.LSTART.setObjectName("LSTART")
        self.LSTOP = QPushButton(parent=self.splitter_14)
        self.LSTOP.setMinimumSize(QSize(116, 26))
        self.LSTOP.setMaximumSize(QSize(116, 26))
        self.LSTOP.setStyleSheet("font: bold 12pt \"Consolas\";")
        self.LSTOP.setCheckable(True)
        self.LSTOP.setChecked(True)
        self.LSTOP.setObjectName("LSTOP")
        self.gridLayout_7.addWidget(self.splitter_14, 0, 0, 1, 1)
        self.label_15 = QLabel(parent=self.splitter_15)
        self.label_15.setMinimumSize(QSize(261, 0))
        self.label_15.setMaximumSize(QSize(261, 16777215))
        self.label_15.setText("")
        self.label_15.setObjectName("label_15")
        self.groupBox_5 = QGroupBox(parent=self.splitter_15)
        self.groupBox_5.setMinimumSize(QSize(261, 121))
        self.groupBox_5.setMaximumSize(QSize(261, 121))
        self.groupBox_5.setStyleSheet("font: 9pt \"Arial\";")
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_8 = QGridLayout(self.groupBox_5)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_7 = QLabel(parent=self.groupBox_5)
        self.label_7.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_8.addWidget(self.label_7, 0, 0, 1, 1)
        self.MRS = QLabel(parent=self.groupBox_5)
        self.MRS.setStyleSheet("font: bold 9pt \"Malgun Gothic\";\n"
                               "color: rgb(255, 0, 0);")
        self.MRS.setObjectName("MRS")
        self.gridLayout_8.addWidget(self.MRS, 0, 1, 1, 1)
        self.label_10 = QLabel(parent=self.groupBox_5)
        self.label_10.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_8.addWidget(self.label_10, 1, 0, 1, 1)
        self.EMGS = QLabel(parent=self.groupBox_5)
        self.EMGS.setStyleSheet("font: bold 9pt \"Malgun Gothic\";\n"
                                "color: rgb(255, 0, 0);")
        self.EMGS.setObjectName("EMGS")
        self.gridLayout_8.addWidget(self.EMGS, 1, 1, 1, 1)
        self.label_8 = QLabel(parent=self.groupBox_5)
        self.label_8.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_8.addWidget(self.label_8, 2, 0, 1, 1)
        self.AIM = QLabel(parent=self.groupBox_5)
        self.AIM.setStyleSheet("font: bold 9pt \"Malgun Gothic\";\n"
                               "color: rgb(255, 0, 0);")
        self.AIM.setObjectName("AIM")
        self.gridLayout_8.addWidget(self.AIM, 2, 1, 1, 1)
        self.label_9 = QLabel(parent=self.groupBox_5)
        self.label_9.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout_8.addWidget(self.label_9, 3, 0, 1, 1)
        self.LS = QLabel(parent=self.groupBox_5)
        self.LS.setStyleSheet("font: bold 9pt \"Malgun Gothic\";\n"
                              "color: rgb(255, 0, 0);")
        self.LS.setObjectName("LS")
        self.gridLayout_8.addWidget(self.LS, 3, 1, 1, 1)
        self.gridLayout_9.addWidget(self.splitter_16, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.AX.setTitle(_translate("MainWindow", "Axis"))
        self.label_3.setText(_translate("MainWindow", "X-Axis"))
        self.label_4.setText(_translate("MainWindow", "Y-Axis"))
        self.GS.setTitle(_translate("MainWindow", "Graph Style"))
        self.CH.setItemText(0, _translate("MainWindow", "CH 1"))
        self.COLOR.setToolTip(_translate("MainWindow", "Line Color"))
        self.COLOR.setText(_translate("MainWindow", "■■■■"))
        self.label_5.setText(_translate("MainWindow", "LineWidth :"))
        self.FILTER.setTitle(_translate("MainWindow", "Filtering"))
        self.F1.setToolTip(_translate("MainWindow", "Original Signal"))
        self.F1.setText(_translate("MainWindow", "ORG"))
        self.F2.setToolTip(_translate("MainWindow", "Average Filter"))
        self.F2.setText(_translate("MainWindow", "AVG"))
        self.F3.setToolTip(_translate("MainWindow", "Savitzky-Golay Filter"))
        self.F3.setText(_translate("MainWindow", "SG"))
        self.F4.setToolTip(_translate("MainWindow", "Average Filter + Savitzky-Golay Filter"))
        self.F4.setText(_translate("MainWindow", "AVG + SG"))
        self.label.setText(_translate("MainWindow", "PORT :"))
        self.OPEN_TCP.setText(_translate("MainWindow", "OPEN"))
        self.CLOSE_TCP.setText(_translate("MainWindow", "CLOSE"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1), _translate("MainWindow", "MR Server"))
        self.label_2.setText(_translate("MainWindow", "PORT :"))
        self.OPEN_EMG.setText(_translate("MainWindow", "OPEN"))
        self.CLOSE_EMG.setText(_translate("MainWindow", "CLOSE"))
        self.DL.setTitle(_translate("MainWindow", "Data Logging"))
        self.LSTART.setToolTip(_translate("MainWindow", "Logging Start"))
        self.LSTART.setText(_translate("MainWindow", "▶"))
        self.LSTOP.setToolTip(_translate("MainWindow", "Logging Stop"))
        self.LSTOP.setText(_translate("MainWindow", "■"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Status"))
        self.label_7.setText(_translate("MainWindow", "MR Server"))
        self.MRS.setText(_translate("MainWindow", "●"))
        self.label_10.setText(_translate("MainWindow", "EMG Sensor"))
        self.EMGS.setText(_translate("MainWindow", "●"))
        self.label_8.setText(_translate("MainWindow", "AI Model"))
        self.AIM.setText(_translate("MainWindow", "●"))
        self.label_9.setText(_translate("MainWindow", "Logging Status"))
        self.LS.setText(_translate("MainWindow", "●"))


class UiMainWindow(QMainWindow, FormWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Viewer")

        self.PLOT.setBackground('l')
        self.PLOT.getAxis('left').setPen('k')
        self.PLOT.getAxis('left').setTextPen('k')
        self.PLOT.getAxis('bottom').setPen('k')
        self.PLOT.getAxis('bottom').setTextPen('k')

        self.PLOT.showGrid(x=True, y=True)
        self.PLOT.enableAutoRange(axis='x')
        self.PLOT.enableAutoRange(axis='y')

        self.x_scale = 50
        self.y_scale = 200

        self.getcolor = "#FF0000"
        self.COLOR.setStyleSheet("QWidget { font: bold 10pt Consolas; color: %s}" % self.getcolor)
        self.getwidth = 1
        self.penColor = mkPen(color=self.getcolor, width=self.getwidth)

        self.isRecord = False
        self.UDP = False
        self.windows_user_name = os.path.expanduser('~')

        self.emg, self.timer_emg, self.time_str, self.f, self.csv_f = None, None, None, None, None
        self.udp_socket, self.udp_host, self.udp_port = None, None, None
        self.box = np.ones(10) / 10

        self.f1, self.f2, self.f3, self.f4 = [], [], [], []
        self.sf1, self.sf2, self.sf3, self.sf4 = [], [], [], []
        self.bar = [0] * 300
        self.Bav = [0]
        self.cls_num = 4

        self.EMG_thread = EMG()
        self.EMG_thread.cls.connect(self.showEMG)
        self.EMG_thread.start()
        self.AIM.setStyleSheet("QWidget { font: bold 9pt Malgun Gothic; color: rgb(0,255,0)}")
        self.AI.setScaledContents(True)

        self.slider_x.valueChanged.connect(self.x_slider)
        self.spin_x.valueChanged.connect(self.x_spin)
        self.slider_y.valueChanged.connect(self.y_slider)
        self.spin_y.valueChanged.connect(self.y_spin)

        self.COLOR.clicked.connect(self.line_color)

        self.OPEN_TCP.clicked.connect(self.udp_socket_open)
        self.CLOSE_TCP.clicked.connect(self.udp_socket_close)

        self.OPEN_EMG.clicked.connect(self.emg_open)
        self.CLOSE_EMG.clicked.connect(self.emg_close)

        self.LSTART.clicked.connect(self.log_start)
        self.LSTOP.clicked.connect(self.log_stop)
        self.LINEWIDTH.valueChanged.connect(self.line_width)

        self.start_time = time.time()  # 시작 시간 재설정
        self.sample_count = 0  # 샘플 개수 초기화

    def emg_open(self):
        try:
            self.emg = Serial('COM' + str(self.PORT_EMG.text()), baudrate=500000)
            if self.emg.readable():
                self.f1 = [int(self.emg.readline().decode().split(',')[1])] * 10
                self.f2 = [sum(self.f1) / 10] * 10
                self.f3 = np.convolve(self.f1, self.box, mode='same')[-1]
                self.f4 = np.convolve(self.f2, self.box, mode='same')[-1]

            self.sf1, self.sf2, self.sf3, self.sf4 = [int(self.f1[0])] * 300, [int(self.f2[0])] * 300, [
                int(self.f3)] * 300, [int(self.f4)] * 300

            del self.f1, self.f2, self.f3, self.f4

            self.timer_emg = QTimer()
            self.timer_emg.timeout.connect(self.draw_emg)
            self.timer_emg.start()
            self.EMGS.setStyleSheet("QWidget { font: bold 9pt Malgun Gothic; color: rgb(0,255,0)}")
        except Exception as e:
            QMessageBox.warning(self, "Warning", str(e))

    def emg_close(self):
        try:
            self.EMGS.setStyleSheet("QWidget { font: bold 9pt Malgun Gothic; color: rgb(255,0,0)}")
            self.emg.close()
            self.timer_emg.stop()
        except Exception as e:
            QMessageBox.warning(self, "Warning", str(e))

    def draw_emg(self):
        try:
            self.PLOT.clear()
            if self.emg.readable():
                value = int(self.emg.readline().decode().split(',')[1])
                self.update_data(value)

                if self.F1.isChecked():
                    self.plot_data(self.sf1)
                    self.current_time = time.time()  # 현재 시간 기록
                    self.elapsed_time = self.current_time - self.start_time  # 경과 시간 계산
                    self.sample_count += 1  # 샘플 개수 증가
                    # 1초가 경과했을 때 샘플링 레이트 출력
                    if self.elapsed_time >= 1:
                        print(f"Samples per second: {self.sample_count}")
                        self.start_time = time.time()  # 시작 시간 재설정
                        self.sample_count = 0  # 샘플 개수 초기화
                elif self.F2.isChecked():
                    self.plot_data(self.sf2)
                elif self.F3.isChecked():
                    self.plot_data(self.sf3)
                elif self.F4.isChecked():
                    self.plot_data(self.sf4)

                # self.send_data(f"{self.sf1[-1]},{self.sf2[-1]},{self.sf3[-1]},{self.sf4[-1]}")
                self.EMG_thread.set_sf4_value(self.sf4[-10:])

                if self.isRecord:
                    self.csv_f.writerow([self.sf1[-1], self.sf2[-1], self.sf3[-1], self.sf4[-1],
                                         datetime.utcnow().strftime('%Y-%m-%d+%H:%M:%S.%f')])
        except Exception as e:
            QMessageBox.warning(self, "Warning", str(e))

    def update_data(self, value):
        self.sf1.append(value)
        self.sf2.append(sum(self.sf1[-10:]) / 10)
        self.sf3.append(np.convolve(self.sf1[-10:], self.box, mode='same')[-1])
        self.sf4.append(np.convolve(self.sf2[-10:], self.box, mode='same')[-1])

        del self.sf1[0], self.sf2[0], self.sf3[0], self.sf4[0]

    def plot_data(self, data):
        self.PLOT.plot(data[-self.x_scale:], pen=self.penColor)

    def showEMG(self, data):
        del self.bar[0]
        self.bar.append(data)
        setdata = self.bar[-1]
        if setdata == 0:
            print('휴식 상태')
        elif setdata == 1:
            print('가위')
        elif setdata == 2:
            print('바위')
        elif setdata == 3:
            print('보')
        self.Bav.append(data)
        if self.Bav[0] != self.Bav[1]:
            datac = str((self.Bav[1]))

            if test_mode:
                if self.Bav[0] != 0 and self.Bav[1] != 0:
                    pass
                else:
                    print(datac)
                    encode_data = str(datac)
                    self.send_data(encode_data)
            else:
                print(datac)
                encode_data = str(datac)
                self.send_data(encode_data)
            del self.Bav[0]
        else:
            del self.Bav[0]

        img = self.bar[-self.x_scale:]
        img = np.array([int(x * (255 / self.cls_num)) for x in img]).astype(np.uint8)
        image = cv2.resize(img, (21, self.AI.width()), interpolation=cv2.INTER_NEAREST)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = QImage(image.data, self.AI.width(), 21, self.AI.width() * 3, QImage.Format.Format_RGB888)
        image = QPixmap.fromImage(image)
        self.AI.setPixmap(image)

    def udp_socket_open(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('', int(self.PORT_TCP.text())))  # 포트 설정 부분
        self.server_socket.listen()

        threading.Thread(target=self.execute).start()
        self.MRS.setText('Waiting...')

    def execute(self):
        try:
            while True:
                self.client_socket, self.addr = self.server_socket.accept()
                self.UDP = True
                print(self.client_socket, self.addr)
                self.MRS.setText('●')
                self.MRS.setStyleSheet("QWidget { font: bold 9pt Malgun Gothic; color: rgb(0,255,0)}")
        except:
            self.MRS.setStyleSheet("QWidget { font: bold 9pt Malgun Gothic; color: rgb(255,0,0)}")
            self.UDP = False
        finally:
            self.UDP = False
            self.server_socket.close()
            self.MRS.setStyleSheet("QWidget { font: bold 9pt Malgun Gothic; color: rgb(255,0,0)}")

    def udp_socket_close(self):
        self.UDP = False
        self.server_socket.close()
        self.MRS.setStyleSheet("QWidget { font: bold 9pt Malgun Gothic; color: rgb(255,0,0)}")

    def send_data(self, data):
        if self.UDP:
            self.client_socket.send(data.encode())

    def x_slider(self):
        self.x_scale = self.slider_x.value()
        self.spin_x.setValue(self.x_scale)

    def x_spin(self):
        self.x_scale = self.spin_x.value()
        self.slider_x.setValue(self.x_scale)

    def y_slider(self):
        self.y_scale = self.slider_y.value()
        self.spin_y.setValue(self.y_scale)
        self.PLOT.setYRange(0, self.y_scale)

    def y_spin(self):
        self.y_scale = self.spin_y.value()
        self.slider_y.setValue(self.y_scale)
        self.PLOT.setYRange(0, self.y_scale)

    def line_color(self):
        self.getcolor = QColorDialog.getColor().name()
        self.COLOR.setStyleSheet("QWidget { font: bold 10pt Consolas; color: %s}" % self.getcolor)
        self.penColor = mkPen(color=self.getcolor, width=self.getwidth)

    def line_width(self):
        self.getwidth = float(self.LINEWIDTH.text())
        self.penColor = mkPen(color=self.getcolor, width=self.getwidth)

    def log_start(self):
        self.LSTOP.setChecked(False)
        self.isRecord = True
        os.makedirs(self.windows_user_name + "/Desktop/Record/", exist_ok=True)
        self.time_str = time.strftime("%Y%m%d_%H%M%S")
        self.f = open(self.windows_user_name + "/Desktop/Record/" + self.time_str + ".csv", "w", encoding='utf-8',
                      newline='')
        self.csv_f = csv.writer(self.f)
        self.LS.setStyleSheet("QWidget { font: bold 9pt Malgun Gothic; color: rgb(0,255,0)}")

    def log_stop(self):
        self.LSTART.setChecked(False)
        self.isRecord = False
        self.f.close()
        self.LS.setStyleSheet("QWidget { font: bold 9pt Malgun Gothic; color: rgb(255,0,0)}")


class EMG(QThread):
    cls = pyqtSignal(int)

    def __init__(self):
        super(EMG, self).__init__()
        self.model, self.device, self.sf4_value = None, None, None
        self.mutex = QMutex()
        self.valuelist = []

    def set_sf4_value(self, value):
        self.mutex.lock()
        self.sf4_value = value
        self.mutex.unlock()
        data = torch.from_numpy(np.reshape(np.array(self.sf4_value, dtype=np.float32), (10, 1))).to(self.device)
        output = self.model(data)
        # self.cls.emit(torch.argmax(output, dim=1).cpu())
        value = self.most_common_value(torch.argmax(output, dim=1).cpu().tolist())
        self.valuelist.append(value)
        cav = self.most_common_value(self.valuelist)
        if len(self.valuelist) == 45:
            self.valuelist = []
        self.cls.emit(cav)

    def most_common_value(self, input_list):
        counter = Counter(input_list)
        return counter.most_common(1)[0][0]

    def run(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.device = torch.device(f'cuda', 0)
        elif torch.backends.mps.is_available():
            self.device = torch.device(f'mps', 0)
        else:
            self.device = torch.device(f'cpu')

        if mode == 1:
            self.model = PModelHJY().to(self.device)
            self.model.load_state_dict(torch.load('weights/hjy.pt'))
        elif mode == 2:
            self.model = PModelHHJ84().to(self.device)
            self.model.load_state_dict(torch.load('weights/hhj0329.pt'))
        elif mode == 3:
            self.model = PModel8114().to(self.device)
            self.model.load_state_dict(torch.load('weights/300.pt'))
        elif mode == 4:
            self.model = PModel8114().to(self.device)
            self.model.load_state_dict(torch.load('weights/400.pt'))
        print('model complete')


class PModelHJY(nn.Module):
    def __init__(self, time_slot: int = 10, depth: int = 4, num_class: int = 4, channel: int = 1):
        super(PModelHJY, self).__init__()
        self.time_slot = time_slot
        self.depth = depth
        self.num_class = num_class
        self.channel = channel
        self.GRU1 = nn.GRU(input_size=self.channel, hidden_size=time_slot, batch_first=True,
                           num_layers=1, bidirectional=False)
        self.GP1 = nn.GRU(input_size=self.channel, hidden_size=1024, batch_first=True,
                           num_layers=1, bidirectional=False)
        self.GP2 = nn.GRU(input_size=1024, hidden_size=512, batch_first=False,
                           num_layers=1, bidirectional=True)
        self.GRU5 = nn.GRU(input_size=time_slot, hidden_size=int(time_slot/2), batch_first=False,
                           num_layers=depth, bidirectional=True)
        self.Dense1000 = nn.Linear(time_slot, 1000)
        self.DenseInput = nn.Linear(1024, 1024)
        self.Dense64 = nn.Linear(1024, time_slot)
        self.DROP = nn.Dropout(0.5)
        self.CLS = nn.Linear(time_slot, num_class)
        self.SOFT = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.GP1(x)
        y = self.DenseInput(x)
        x_d1 = self.DROP(x)
        x2, _ = self.GP2(x_d1)
        x_d2 = self.DROP(x)
        x_den = x_d2 + y
        x_den = self.Dense64(x_den)
        x_cls = self.CLS(x_den)
        return x_cls


class PModelHHJ84(nn.Module):
    def __init__(self, time_slot: int = 10, depth: int = 8, num_class: int = 4, channel: int = 1):
        super(PModelHHJ84, self).__init__()
        self.time_slot = time_slot
        self.depth = depth
        self.num_class = num_class
        self.channel = channel
        self.GRU1 = nn.GRU(input_size=self.channel, hidden_size=time_slot, batch_first=True,
                           num_layers=1, bidirectional=False)
        self.GRU5 = nn.GRU(input_size=time_slot, hidden_size=int(time_slot / 2), batch_first=False,
                           num_layers=depth, bidirectional=True)
        self.Dense1000 = nn.Linear(time_slot, 1000)
        self.Dense100 = nn.Linear(time_slot, 100)
        self.DenseInput = nn.Linear(1024, 1024)
        self.Dense64 = nn.Linear(1000, time_slot)
        self.Dense10 = nn.Linear(100, time_slot)
        self.DROP = nn.Dropout(0.5)
        self.CLS = nn.Linear(time_slot, num_class)
        self.SOFT = nn.Softmax(dim=2)

    def forward(self, x):
        x, _ = self.GRU1(x)
        x_gru, _ = self.GRU5(x)
        x_den = self.Dense1000(x_gru)
        x_den = self.DROP(x_den)
        x_den = self.Dense64(x_den)
        x_cls = self.CLS(x_den)
        return x_cls


class PModel8114(nn.Module):
    def __init__(self, time_slot: int = 10, depth: int = 8, num_class: int = 4, channel: int = 1):
        super(PModel8114, self).__init__()
        self.time_slot = time_slot
        self.depth = depth
        self.num_class = num_class
        self.channel = channel
        self.GRU1 = nn.LSTM(input_size=self.channel, hidden_size=time_slot, batch_first=True,
                           num_layers=1, bidirectional=False)
        self.GRU5 = nn.LSTM(input_size=time_slot, hidden_size=int(time_slot/2), batch_first=False,
                           num_layers=depth, bidirectional=True)
        self.Dense100 = nn.Linear(time_slot, 100)
        self.Dense10 = nn.Linear(100, time_slot)
        self.DROP = nn.Dropout(0.7)
        self.CLS = nn.Linear(time_slot, num_class)
        self.SOFT = nn.Softmax(dim=2)

    def forward(self, x):
        x,_ = self.GRU1(x)
        x_gru, _ = self.GRU5(x)
        x_den = self.Dense100(x_gru)
        x_den = self.DROP(x_den)
        x_den = self.Dense10(x_den)
        x_cls = self.CLS(x_den)
        return x_cls


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = UiMainWindow()
    MainWindow.show()
    sys.exit(app.exec())