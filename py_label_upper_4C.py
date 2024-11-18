import csv
import os
import sys
import time
from datetime import datetime
import random

import numpy as np
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from pyqtgraph import PlotWidget, mkPen
from serial import Serial


class FormWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(817, 519)
        MainWindow.setMinimumSize(QSize(817, 519))
        self.centralwidget = QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_9 = QGridLayout(self.centralwidget)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.splitter_12 = QSplitter(parent=self.centralwidget)
        self.splitter_12.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_12.setObjectName("splitter_12")
        self.splitter_6 = QSplitter(parent=self.splitter_12)
        self.splitter_6.setMinimumSize(QSize(531, 501))
        self.splitter_6.setOrientation(Qt.Orientation.Vertical)
        self.splitter_6.setObjectName("splitter_6")
        self.splitter_5 = QSplitter(parent=self.splitter_6)
        self.splitter_5.setMinimumSize(QSize(531, 401))
        self.splitter_5.setMaximumSize(QSize(16777215, 16777215))
        self.splitter_5.setOrientation(Qt.Orientation.Vertical)
        self.splitter_5.setObjectName("splitter_5")
        self.splitter = QSplitter(parent=self.splitter_5)
        self.splitter.setMinimumSize(QSize(521, 21))
        self.splitter.setMaximumSize(QSize(16777215, 21))
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("splitter")
        self.label_8 = QLabel(parent=self.splitter)
        self.label_8.setMinimumSize(QSize(0, 21))
        self.label_8.setMaximumSize(QSize(16777215, 21))
        self.label_8.setStyleSheet("font: 9pt \"Arial\";")
        self.label_8.setFrameShape(QFrame.Shape.NoFrame)
        self.label_8.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.CC = QLabel(parent=self.splitter)
        self.CC.setMinimumSize(QSize(0, 21))
        self.CC.setMaximumSize(QSize(16777215, 21))
        self.CC.setStyleSheet("font: bold 9pt \"Arial\";")
        self.CC.setFrameShape(QFrame.Shape.NoFrame)
        self.CC.setObjectName("CC")
        self.label_11 = QLabel(parent=self.splitter)
        self.label_11.setMinimumSize(QSize(0, 21))
        self.label_11.setMaximumSize(QSize(16777215, 21))
        self.label_11.setStyleSheet("font: 9pt \"Arial\";")
        self.label_11.setFrameShape(QFrame.Shape.NoFrame)
        self.label_11.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.label_11.setObjectName("label_11")
        self.CT = QLabel(parent=self.splitter)
        self.CT.setMinimumSize(QSize(0, 21))
        self.CT.setMaximumSize(QSize(16777215, 21))
        self.CT.setStyleSheet("font: bold 9pt \"Arial\";")
        self.CT.setFrameShape(QFrame.Shape.NoFrame)
        self.CT.setObjectName("CT")
        self.PLOT = PlotWidget(parent=self.splitter_5)
        self.PLOT.setMinimumSize(QSize(521, 371))
        self.PLOT.setFrameShape(QFrame.Shape.Box)
        self.PLOT.setFrameShadow(QFrame.Shadow.Plain)
        self.PLOT.setObjectName("PLOT")
        self.splitter_9 = QSplitter(parent=self.splitter_6)
        self.splitter_9.setMinimumSize(QSize(531, 90))
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
        self.label_3.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.slider_x = QSlider(parent=self.splitter_7)
        self.slider_x.setMinimum(50)
        self.slider_x.setMaximum(2000)
        self.slider_x.setProperty("value", 50)
        self.slider_x.setOrientation(Qt.Orientation.Horizontal)
        self.slider_x.setObjectName("slider_x")
        self.spin_x = QSpinBox(parent=self.splitter_7)
        self.spin_x.setMinimumSize(QSize(50, 22))
        self.spin_x.setMaximumSize(QSize(50, 22))
        self.spin_x.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin_x.setMinimum(50)
        self.spin_x.setMaximum(2000)
        self.spin_x.setDisplayIntegerBase(10)
        self.spin_x.setObjectName("spin_x")
        self.gridLayout_3.addWidget(self.splitter_7, 0, 0, 1, 1)
        self.splitter_8 = QSplitter(parent=self.AX)
        self.splitter_8.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_8.setObjectName("splitter_8")
        self.label_4 = QLabel(parent=self.splitter_8)
        self.label_4.setMinimumSize(QSize(40, 22))
        self.label_4.setMaximumSize(QSize(40, 22))
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.slider_y = QSlider(parent=self.splitter_8)
        self.slider_y.setMinimum(10)
        self.slider_y.setMaximum(200)
        self.slider_y.setProperty("value", 200)
        self.slider_y.setOrientation(Qt.Orientation.Horizontal)
        self.slider_y.setObjectName("slider_y")
        self.spin_y = QSpinBox(parent=self.splitter_8)
        self.spin_y.setMinimumSize(QSize(50, 22))
        self.spin_y.setMaximumSize(QSize(50, 22))
        self.spin_y.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin_y.setMinimum(10)
        self.spin_y.setMaximum(200)
        self.spin_y.setProperty("value", 200)
        self.spin_y.setObjectName("spin_y")
        self.gridLayout_3.addWidget(self.splitter_8, 1, 0, 1, 1)
        self.GS = QGroupBox(parent=self.splitter_9)
        self.GS.setMinimumSize(QSize(141, 90))
        self.GS.setMaximumSize(QSize(141, 90))
        self.GS.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.GS.setStyleSheet("font: 9pt \"Arial\";")
        self.GS.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
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
        self.CH.addItems(["CH 1", "CH 2", "CH 3", "CH 4"])
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
        self.FILTER.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
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
        self.splitter_4 = QSplitter(parent=self.splitter_12)
        self.splitter_4.setOrientation(Qt.Orientation.Vertical)
        self.splitter_4.setObjectName("splitter_4")
        self.groupBox = QGroupBox(parent=self.splitter_4)
        self.groupBox.setMinimumSize(QSize(261, 91))
        self.groupBox.setMaximumSize(QSize(261, 91))
        self.groupBox.setStyleSheet("font: 9pt \"Arial\";")
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QLabel(parent=self.groupBox)
        self.label_2.setMinimumSize(QSize(116, 24))
        self.label_2.setMaximumSize(QSize(116, 24))
        self.label_2.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.label_2.setStyleSheet("font: bold 9pt \"Arial\";")
        self.label_2.setTextFormat(Qt.TextFormat.AutoText)
        self.label_2.setScaledContents(False)
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.PORT_EMG = QLineEdit(parent=self.groupBox)
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
        self.OPEN_EMG = QPushButton(parent=self.groupBox)
        self.OPEN_EMG.setMinimumSize(QSize(116, 26))
        self.OPEN_EMG.setMaximumSize(QSize(116, 26))
        self.OPEN_EMG.setAutoFillBackground(False)
        self.OPEN_EMG.setAutoRepeat(False)
        self.OPEN_EMG.setDefault(False)
        self.OPEN_EMG.setFlat(False)
        self.OPEN_EMG.setObjectName("OPEN_EMG")
        self.gridLayout.addWidget(self.OPEN_EMG, 1, 0, 1, 1)
        self.CLOSE_EMG = QPushButton(parent=self.groupBox)
        self.CLOSE_EMG.setMinimumSize(QSize(116, 26))
        self.CLOSE_EMG.setMaximumSize(QSize(116, 26))
        self.CLOSE_EMG.setObjectName("CLOSE_EMG")
        self.gridLayout.addWidget(self.CLOSE_EMG, 1, 1, 1, 1)
        self.groupBox_2 = QGroupBox(parent=self.splitter_4)
        self.groupBox_2.setMinimumSize(QSize(261, 111))
        self.groupBox_2.setMaximumSize(QSize(261, 111))
        self.groupBox_2.setStyleSheet("font: 9pt \"Arial\";")
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QLabel(parent=self.groupBox_2)
        self.label.setMinimumSize(QSize(81, 22))
        self.label.setMaximumSize(QSize(81, 22))
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.RT = QLineEdit(parent=self.groupBox_2)
        self.RT.setObjectName("RT")
        self.gridLayout_2.addWidget(self.RT, 0, 1, 1, 1)
        self.label_6 = QLabel(parent=self.groupBox_2)
        self.label_6.setMinimumSize(QSize(81, 22))
        self.label_6.setMaximumSize(QSize(81, 22))
        self.label_6.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 1, 0, 1, 1)
        self.AT = QLineEdit(parent=self.groupBox_2)
        self.AT.setObjectName("AT")
        self.gridLayout_2.addWidget(self.AT, 1, 1, 1, 1)
        self.label_7 = QLabel(parent=self.groupBox_2)
        self.label_7.setMinimumSize(QSize(81, 22))
        self.label_7.setMaximumSize(QSize(81, 22))
        self.label_7.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 2, 0, 1, 1)
        self.CN = QComboBox(parent=self.groupBox_2)
        self.CN.setObjectName("CN")
        self.CN.addItem("")
        self.CN.addItem("")
        self.CN.addItem("")
        self.CN.addItem("")
        self.CN.addItem("")
        self.CN.addItem("")
        self.CN.addItem("")
        self.CN.addItem("")
        self.CN.addItem("")
        self.gridLayout_2.addWidget(self.CN, 2, 1, 1, 1)
        self.DL = QGroupBox(parent=self.splitter_4)
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
        self.splitter_2 = QSplitter(parent=self.splitter_4)
        self.splitter_2.setMinimumSize(QSize(261, 21))
        self.splitter_2.setMaximumSize(QSize(261, 21))
        self.splitter_2.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.label_12 = QLabel(parent=self.splitter_2)
        self.label_12.setMinimumSize(QSize(75, 21))
        self.label_12.setMaximumSize(QSize(75, 21))
        self.label_12.setStyleSheet("font: 9pt \"Arial\";")
        self.label_12.setFrameShape(QFrame.Shape.NoFrame)
        self.label_12.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.label_13 = QLabel(parent=self.splitter_2)
        self.label_13.setMinimumSize(QSize(181, 21))
        self.label_13.setMaximumSize(QSize(181, 21))
        self.label_13.setStyleSheet("font: 9pt \"Arial\";")
        self.label_13.setFrameShape(QFrame.Shape.NoFrame)
        self.label_13.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.splitter_3 = QSplitter(parent=self.splitter_4)
        self.splitter_3.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_3.setObjectName("splitter_3")
        self.CLABEL = QLabel(parent=self.splitter_3)
        self.CLABEL.setMinimumSize(QSize(75, 75))
        self.CLABEL.setMaximumSize(QSize(75, 75))
        self.CLABEL.setStyleSheet("font: bold 32pt \"Arial\";")
        self.CLABEL.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.CLABEL.setObjectName("CLABEL")
        self.TLABEL = QLabel(parent=self.splitter_3)
        self.TLABEL.setMinimumSize(QSize(181, 75))
        self.TLABEL.setMaximumSize(QSize(181, 75))
        self.TLABEL.setStyleSheet("font: bold 32pt \"Arial\";")
        self.TLABEL.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.TLABEL.setObjectName("TLABEL")
        self.label_15 = QLabel(parent=self.splitter_4)
        self.label_15.setMinimumSize(QSize(261, 0))
        self.label_15.setMaximumSize(QSize(261, 16777215))
        self.label_15.setText("")
        self.label_15.setObjectName("label_15")
        self.groupBox_5 = QGroupBox(parent=self.splitter_4)
        self.groupBox_5.setMinimumSize(QSize(261, 91))
        self.groupBox_5.setMaximumSize(QSize(261, 91))
        self.groupBox_5.setStyleSheet("font: 9pt \"Arial\";")
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_8 = QGridLayout(self.groupBox_5)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.EMGS = QLabel(parent=self.groupBox_5)
        self.EMGS.setStyleSheet("font: bold 9pt \"Malgun Gothic\";\n"
"color: rgb(255, 0, 0);")
        self.EMGS.setObjectName("EMGS")
        self.gridLayout_8.addWidget(self.EMGS, 0, 1, 1, 1)
        self.LS = QLabel(parent=self.groupBox_5)
        self.LS.setStyleSheet("font: bold 9pt \"Malgun Gothic\";\n"
"color: rgb(255, 0, 0);")
        self.LS.setObjectName("LS")
        self.gridLayout_8.addWidget(self.LS, 1, 1, 1, 1)
        self.label_10 = QLabel(parent=self.groupBox_5)
        self.label_10.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_8.addWidget(self.label_10, 0, 0, 1, 1)
        self.label_9 = QLabel(parent=self.groupBox_5)
        self.label_9.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout_8.addWidget(self.label_9, 1, 0, 1, 1)
        self.gridLayout_9.addWidget(self.splitter_12, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_8.setText(_translate("MainWindow", "Class"))
        self.CC.setText(_translate("MainWindow", "0"))
        self.label_11.setText(_translate("MainWindow", "Rem. Time (s)"))
        self.CT.setText(_translate("MainWindow", "0.0"))
        self.AX.setTitle(_translate("MainWindow", "Axis"))
        self.label_3.setText(_translate("MainWindow", "X-Axis"))
        self.label_4.setText(_translate("MainWindow", "Y-Axis"))
        self.GS.setTitle(_translate("MainWindow", "Graph Style"))
        self.CH.setItemText(0, _translate("MainWindow", "CH 1"))
        self.CH.setItemText(1, _translate("MainWindow", "CH 2"))
        self.CH.setItemText(2, _translate("MainWindow", "CH 3"))
        self.CH.setItemText(3, _translate("MainWindow", "CH 4"))
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
        self.groupBox.setTitle(_translate("MainWindow", "EMG Sensor"))
        self.label_2.setText(_translate("MainWindow", "PORT :"))
        self.OPEN_EMG.setText(_translate("MainWindow", "OPEN"))
        self.CLOSE_EMG.setText(_translate("MainWindow", "CLOSE"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Labeling Setting"))
        self.label.setText(_translate("MainWindow", "Rest Time (s)"))
        self.RT.setText(_translate("MainWindow", "5"))
        self.label_6.setText(_translate("MainWindow", "Active Time (s)"))
        self.AT.setText(_translate("MainWindow", "5"))
        self.label_7.setText(_translate("MainWindow", "Class num"))
        self.CN.setItemText(0, _translate("MainWindow", "2"))
        self.CN.setItemText(1, _translate("MainWindow", "3"))
        self.CN.setItemText(2, _translate("MainWindow", "4"))
        self.CN.setItemText(3, _translate("MainWindow", "5"))
        self.CN.setItemText(4, _translate("MainWindow", "6"))
        self.CN.setItemText(5, _translate("MainWindow", "7"))
        self.CN.setItemText(6, _translate("MainWindow", "8"))
        self.CN.setItemText(7, _translate("MainWindow", "9"))
        self.CN.setItemText(8, _translate("MainWindow", "10"))
        self.DL.setTitle(_translate("MainWindow", "Data Logging"))
        self.LSTART.setToolTip(_translate("MainWindow", "Logging Start"))
        self.LSTART.setText(_translate("MainWindow", "▶"))
        self.LSTOP.setToolTip(_translate("MainWindow", "Logging Stop"))
        self.LSTOP.setText(_translate("MainWindow", "■"))
        self.label_12.setText(_translate("MainWindow", "Class"))
        self.label_13.setText(_translate("MainWindow", "Rem. Time (s)"))
        self.CLABEL.setText(_translate("MainWindow", "X"))
        self.TLABEL.setText(_translate("MainWindow", "0.00"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Status"))
        self.EMGS.setText(_translate("MainWindow", "●"))
        self.LS.setText(_translate("MainWindow", "●"))
        self.label_10.setText(_translate("MainWindow", "EMG Sensor"))
        self.label_9.setText(_translate("MainWindow", "Logging Status"))


class UiMainWindow(QMainWindow, FormWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Viewer")

        custom_color = QColor(200, 160, 160)  # Example: a custom RGB color
        self.PLOT.setBackground(custom_color)
        self.PLOT.getAxis('left').setPen('k')
        self.PLOT.getAxis('left').setTextPen('k')
        self.PLOT.getAxis('bottom').setPen('k')
        self.PLOT.getAxis('bottom').setTextPen('k')

        self.PLOT.showGrid(x=True, y=True)
        self.PLOT.enableAutoRange(axis='x')
        self.PLOT.enableAutoRange(axis='y')

        self.x_scale = 50
        self.y_scale = 200
        self.record_time = 300

        self.getcolor_C1 = "#FF0000"  # 빨간색
        self.getcolor_C2 = "#00FF00"  # 초록색
        self.getcolor_C3 = "#0000FF"  # 파란색
        self.getcolor_C4 = "#FFFF00"  # 노란색

        self.getwidth = 1
        # 채널 1에 대한 설정
        self.COLOR.setStyleSheet("QWidget { font: bold 10pt Consolas; color: %s}" % self.getcolor_C1)
        self.penColor_C1 = mkPen(color=self.getcolor_C1, width=self.getwidth)

        # 채널 2에 대한 설정
        self.COLOR.setStyleSheet("QWidget { font: bold 10pt Consolas; color: %s}" % self.getcolor_C2)
        self.penColor_C2 = mkPen(color=self.getcolor_C2, width=self.getwidth)

        # 채널 3에 대한 설정
        self.COLOR.setStyleSheet("QWidget { font: bold 10pt Consolas; color: %s}" % self.getcolor_C3)
        self.penColor_C3 = mkPen(color=self.getcolor_C3, width=self.getwidth)

        # 채널 4에 대한 설정
        self.COLOR.setStyleSheet("QWidget { font: bold 10pt Consolas; color: %s}" % self.getcolor_C4)
        self.penColor_C4 = mkPen(color=self.getcolor_C4, width=self.getwidth)


        self.isRecord = False
        self.windows_user_name = os.path.expanduser('~')

        self.emg, self.timer_emg, self.time_str, self.f, self.csv_f = None, None, None, None, None
        self.box = np.ones(10) / 10

        self.C1_f1, self.C1_f2, self.C1_f3, self.C1_f4 = [], [], [], []
        self.C2_f1, self.C2_f2, self.C2_f3, self.C2_f4 = [], [], [], []
        self.C3_f1, self.C3_f2, self.C3_f3, self.C3_f4 = [], [], [], []
        self.C4_f1, self.C4_f2, self.C4_f3, self.C4_f4 = [], [], [], []
        self.C1_sf1, self.C1_sf2, self.C1_sf3, self.C1_sf4 = [], [], [], []
        self.C2_sf1, self.C2_sf2, self.C2_sf3, self.C2_sf4 = [], [], [], []
        self.C3_sf1, self.C3_sf2, self.C3_sf3, self.C3_sf4 = [], [], [], []
        self.C4_sf1, self.C4_sf2, self.C4_sf3, self.C4_sf4 = [], [], [], []

        self.slider_x.valueChanged.connect(self.x_slider)
        self.spin_x.valueChanged.connect(self.x_spin)
        self.slider_y.valueChanged.connect(self.y_slider)
        self.spin_y.valueChanged.connect(self.y_spin)

        self.COLOR.clicked.connect(self.line_color)

        self.OPEN_EMG.clicked.connect(self.emg_open)
        self.CLOSE_EMG.clicked.connect(self.emg_close)

        self.LSTART.clicked.connect(self.log_start)
        self.LSTOP.clicked.connect(self.log_stop)
        self.LINEWIDTH.valueChanged.connect(self.line_width)

        self.start_time = time.time()  # 시작 시간 재설정
        self.sample_count = 0  # 샘플 개수 초기화
        self.amplification_factor = 20.0  # 원하는 증폭 배율을 설정합니다.


    def emg_open(self):
        try:
            self.emg = Serial('COM'+str(self.PORT_EMG.text()), baudrate=9600)
            if self.emg.readable():
                self.C1_f1 = [int(self.emg.readline().decode().split(',')[0]) * self.amplification_factor - 2000] * 10
                self.C2_f1 = [int(self.emg.readline().decode().split(',')[1]) * self.amplification_factor - 2000] * 10
                self.C3_f1 = [int(self.emg.readline().decode().split(',')[2]) * self.amplification_factor - 2000] * 10
                self.C4_f1 = [int(self.emg.readline().decode().split(',')[3]) * self.amplification_factor - 2000] * 10
                self.C1_f2 = [sum(self.C1_f1)/10] * 10
                self.C2_f2 = [sum(self.C2_f1) / 10] * 10
                self.C3_f2 = [sum(self.C3_f1) / 10] * 10
                self.C4_f2 = [sum(self.C4_f1) / 10] * 10
                self.C1_f3 = np.convolve(self.C1_f1, self.box, mode='same')[-1]
                self.C2_f3 = np.convolve(self.C2_f1, self.box, mode='same')[-1]
                self.C3_f3 = np.convolve(self.C3_f1, self.box, mode='same')[-1]
                self.C4_f3 = np.convolve(self.C4_f1, self.box, mode='same')[-1]
                self.C1_f4 = np.convolve(self.C1_f2, self.box, mode='same')[-1]
                self.C2_f4 = np.convolve(self.C2_f2, self.box, mode='same')[-1]
                self.C3_f4 = np.convolve(self.C3_f2, self.box, mode='same')[-1]
                self.C4_f4 = np.convolve(self.C4_f2, self.box, mode='same')[-1]

            self.C1_sf1, self.C1_sf2, self.C1_sf3, self.C1_sf4 = [int(self.C1_f1[0])] * 2000, [int(self.C1_f2[0])] * 2000, [int(self.C1_f3)] * 2000, [int(self.C1_f4)] * 2000
            self.C2_sf1, self.C2_sf2, self.C2_sf3, self.C2_sf4 = [int(self.C2_f1[0])] * 2000, [int(self.C2_f2[0])] * 2000, [int(self.C2_f3)] * 2000, [int(self.C2_f4)] * 2000
            self.C3_sf1, self.C3_sf2, self.C3_sf3, self.C3_sf4 = [int(self.C3_f1[0])] * 2000, [int(self.C3_f2[0])] * 2000, [int(self.C3_f3)] * 2000, [int(self.C3_f4)] * 2000
            self.C4_sf1, self.C4_sf2, self.C4_sf3, self.C4_sf4 = [int(self.C4_f1[0])] * 2000, [int(self.C4_f2[0])] * 2000, [int(self.C4_f3)] * 2000, [int(self.C4_f4)] * 2000


            del self.C1_f1, self.C1_f2, self.C1_f3, self.C1_f4
            del self.C2_f1, self.C2_f2, self.C2_f3, self.C2_f4
            del self.C3_f1, self.C3_f2, self.C3_f3, self.C3_f4
            del self.C4_f1, self.C4_f2, self.C4_f3, self.C4_f4

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
                C1_value = self.emg.readline().decode().split(',')[0]
                C2_value = self.emg.readline().decode().split(',')[1]
                C3_value = self.emg.readline().decode().split(',')[2]
                C4_value = self.emg.readline().decode().split(',')[3]
                if len(C1_value) > 0:
                    self.update_data_C1(int(C1_value))
                    self.update_data_C2(int(C2_value))
                    self.update_data_C3(int(C3_value))
                    self.update_data_C4(int(C4_value))

                    if self.F1.isChecked():
                        self.plot_data_C1(self.C1_sf1)
                        self.plot_data_C2(self.C2_sf1)
                        self.plot_data_C3(self.C3_sf1)
                        self.plot_data_C4(self.C4_sf1)
                        self.current_time = time.time()  # 현재 시간 기록
                        self.elapsed_time = self.current_time - self.start_time  # 경과 시간 계산
                        self.sample_count += 1  # 샘플 개수 증가
                        # 1초가 경과했을 때 샘플링 레이트 출력
                        if self.elapsed_time >= 1:
                            print(f"Samples per second: {self.sample_count}")
                            self.start_time = time.time()  # 시작 시간 재설정
                            self.sample_count = 0  # 샘플 개수 초기화
                    elif self.F2.isChecked():
                        self.plot_data_C1(self.C1_sf2)
                        self.plot_data_C2(self.C2_sf2)
                        self.plot_data_C3(self.C3_sf2)
                        self.plot_data_C4(self.C4_sf2)
                    elif self.F3.isChecked():
                        self.plot_data_C1(self.C1_sf3)
                        self.plot_data_C2(self.C2_sf3)
                        self.plot_data_C3(self.C3_sf3)
                        self.plot_data_C4(self.C4_sf3)
                    elif self.F4.isChecked():
                        self.plot_data_C1(self.C1_sf4)
                        self.plot_data_C2(self.C2_sf4)
                        self.plot_data_C3(self.C3_sf4)
                        self.plot_data_C4(self.C4_sf4)

                    if self.isRecord:
                        if self.t_remain == 0:
                            if self.c_label == 0:
                                self.c_label = self.remaining_labels.pop(0)  # 미리 생성된 레이블 값 중 하나 선택
                                self.t_remain = self.a_time * 77  # active time 설정
                                self.v_time = time.time() + self.a_time
                            else:
                                self.c_label = 0  # rest label 선택
                                self.t_remain = self.r_time * 77  # rest time 설정
                                self.v_time = time.time() + self.r_time

                        self.t_remain -= 1  # 시간 감소
                        self.csv_f.writerow([self.C1_sf2[-1], self.C2_sf2[-1], self.C3_sf2[-1], self.C4_sf2[-1], self.c_label,
                                             datetime.utcnow().strftime('%Y-%m-%d+%H:%M:%S.%f')])
                        rt = round(self.v_time - time.time(), 2)
                        tt = round(self.t_time - time.time(), 2)
                        if self.c_label == 0:
                            self.CLABEL.setText('휴식')
                        elif self.c_label == 1:
                            self.CLABEL.setText('👌')
                        elif self.c_label == 2:
                            self.CLABEL.setText('✊')
                        elif self.c_label == 3:
                            self.CLABEL.setText('✋')
                        self.TLABEL.setText(str(rt if rt > 0 else 0).zfill(2))
                        self.CC.setText(str(self.c_label))
                        self.CT.setText(str(tt if tt > 0 else 0).zfill(2))
                        if tt <= 0:
                            self.LSTOP.setChecked(True)
                            self.log_stop()


        except Exception as e:
            QMessageBox.warning(self, "Warning", str(e))

    def update_data_C1(self, C1_value):
        amplified_value = C1_value * self.amplification_factor - 2000  # 증폭 배율 적용
        self.C1_sf1.append(amplified_value)
        self.C1_sf2.append(sum(self.C1_sf1[-10:]) / 10)
        self.C1_sf3.append(np.convolve(self.C1_sf1[-10:], self.box, mode='same')[-1])
        self.C1_sf4.append(np.convolve(self.C1_sf2[-10:], self.box, mode='same')[-1])

        del self.C1_sf1[0], self.C1_sf2[0], self.C1_sf3[0], self.C1_sf4[0]

    def update_data_C2(self, C2_value):
        amplified_value = C2_value * self.amplification_factor - 2000
        self.C2_sf1.append(amplified_value)
        self.C2_sf2.append(sum(self.C2_sf1[-10:]) / 10)
        self.C2_sf3.append(np.convolve(self.C2_sf1[-10:], self.box, mode='same')[-1])
        self.C2_sf4.append(np.convolve(self.C2_sf2[-10:], self.box, mode='same')[-1])

        del self.C2_sf1[0], self.C2_sf2[0], self.C2_sf3[0], self.C2_sf4[0]

    def update_data_C3(self, C3_value):
        amplified_value = C3_value * self.amplification_factor - 2000
        self.C3_sf1.append(amplified_value)
        self.C3_sf2.append(sum(self.C3_sf1[-10:]) / 10)
        self.C3_sf3.append(np.convolve(self.C3_sf1[-10:], self.box, mode='same')[-1])
        self.C3_sf4.append(np.convolve(self.C3_sf2[-10:], self.box, mode='same')[-1])

        del self.C3_sf1[0], self.C3_sf2[0], self.C3_sf3[0], self.C3_sf4[0]

    def update_data_C4(self, C4_value):
        amplified_value = C4_value * self.amplification_factor - 2000
        self.C4_sf1.append(amplified_value)
        self.C4_sf2.append(sum(self.C4_sf1[-10:]) / 10)
        self.C4_sf3.append(np.convolve(self.C4_sf1[-10:], self.box, mode='same')[-1])
        self.C4_sf4.append(np.convolve(self.C4_sf2[-10:], self.box, mode='same')[-1])

        del self.C4_sf1[0], self.C4_sf2[0], self.C4_sf3[0], self.C4_sf4[0]

    def plot_data_C1(self, data):
        self.PLOT.plot(data[-self.x_scale:], pen=self.penColor_C1)

    def plot_data_C2(self, data):
        self.PLOT.plot(data[-self.x_scale:], pen=self.penColor_C2)

    def plot_data_C3(self, data):
        self.PLOT.plot(data[-self.x_scale:], pen=self.penColor_C3)

    def plot_data_C4(self, data):
        self.PLOT.plot(data[-self.x_scale:], pen=self.penColor_C4)

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
        self.r_time = int(self.RT.text())
        self.a_time = int(self.AT.text())
        c_num = int(self.CN.currentIndex())+1
        r_len = self.record_time * 77
        self.l_counts = [r_len // (c_num + 1)] * (c_num + 1)
        self.remaining_labels = []
        for i in range(1, c_num + 1):
            self.remaining_labels += [i] * self.l_counts[i]  # 레이블 출현 횟수에 맞게 리스트 생성
        self.remaining_labels += [0] * self.l_counts[0]
        random.shuffle(self.remaining_labels)  # 레이블 순서를 무작위로 섞음
        self.t_remain = 0
        self.c_label = 0

        os.makedirs(self.windows_user_name + "/Desktop/Record/", exist_ok=True)
        self.time_str = time.strftime("%Y%m%d_%H%M%S")
        self.t_time = time.time() + 300
        self.f = open(self.windows_user_name + "/Desktop/Record/" + self.time_str + ".csv", "w", encoding='utf-8', newline='')
        self.csv_f = csv.writer(self.f)
        self.LS.setStyleSheet("QWidget { font: bold 9pt Malgun Gothic; color: rgb(0,255,0)}")

    def log_stop(self):
        self.LSTART.setChecked(False)
        self.isRecord = False
        self.f.close()
        self.LS.setStyleSheet("QWidget { font: bold 9pt Malgun Gothic; color: rgb(255,0,0)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = UiMainWindow()
    MainWindow.show()
    sys.exit(app.exec())
