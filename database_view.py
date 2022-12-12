# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'database_view.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import os
from PIL import Image

class Ui_databaseView(object):
    def setupUi(self, databaseView):
        databaseView.setObjectName("databaseView")
        databaseView.resize(433, 418)
        databaseView.setMaximumSize(QtCore.QSize(1106, 700))
        databaseView.setStyleSheet("background: rgb(20,21,38);\n"
        "color: white;")
        self.horizontalLayout = QtWidgets.QHBoxLayout(databaseView)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.img_frame = QtWidgets.QFrame(databaseView)
        self.img_frame.setMaximumSize(QtCore.QSize(700, 700))
        self.img_frame.setStyleSheet("border: 0px;")
        self.img_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.img_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.img_frame.setObjectName("img_frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.img_frame)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 5)
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.img_display = QtWidgets.QLabel(self.img_frame)
        self.img_display.setMaximumSize(QtCore.QSize(700, 700))
        self.img_display.setText("")
        self.img_display.setScaledContents(True)
        self.img_display.setObjectName("img_display")
        self.verticalLayout_2.addWidget(self.img_display)
        self.name_display = QtWidgets.QLabel(self.img_frame)
        self.name_display.setMinimumSize(QtCore.QSize(0, 30))
        self.name_display.setMaximumSize(QtCore.QSize(700, 30))
        self.name_display.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.name_display.setStyleSheet("background:  transparent;\n"
"border : 0px;\n"
"color:    rgb(28,191,255);\n"
"font: 81 bold 19pt \"Cantarell\";")
        self.name_display.setLineWidth(0)
        self.name_display.setText("")
        self.name_display.setAlignment(QtCore.Qt.AlignCenter)
        self.name_display.setObjectName("name_display")
        self.verticalLayout_2.addWidget(self.name_display)
        self.horizontalLayout.addWidget(self.img_frame)
        self.menu_frame = QtWidgets.QFrame(databaseView)
        self.menu_frame.setMinimumSize(QtCore.QSize(400, 400))
        self.menu_frame.setMaximumSize(QtCore.QSize(400, 400))
        self.menu_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.menu_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.menu_frame.setObjectName("menu_frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.menu_frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_3 = QtWidgets.QFrame(self.menu_frame)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bt_load = QtWidgets.QPushButton(self.frame_3)
        self.bt_load.clicked.connect(self.browseFile)
        self.bt_load.setMinimumSize(QtCore.QSize(360, 40))
        self.bt_load.setMaximumSize(QtCore.QSize(360, 40))
        self.bt_load.setStyleSheet("QPushButton:hover{\n"
"border: 5px solid rgb(26,169,253);\n"
"background:rgb(110,120,148);\n"
"border-radius: 15px;\n"
"}")
        self.bt_load.setObjectName("bt_load")
        self.horizontalLayout_2.addWidget(self.bt_load)
        self.verticalLayout.addWidget(self.frame_3)
        self.frame_4 = QtWidgets.QFrame(self.menu_frame)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.bt_search = QtWidgets.QPushButton(self.frame_4)
        self.bt_search.clicked.connect(self.encodeImage)
        self.bt_search.setMinimumSize(QtCore.QSize(360, 40))
        self.bt_search.setMaximumSize(QtCore.QSize(360, 40))
        self.bt_search.setStyleSheet("QPushButton:hover{\n"
"border: 5px solid rgb(26,169,253);\n"
"background:rgb(110,120,148);\n"
"border-radius: 15px;\n"
"}")
        self.bt_search.setObjectName("bt_search")
        self.horizontalLayout_3.addWidget(self.bt_search)
        self.verticalLayout.addWidget(self.frame_4)
        self.frame_5 = QtWidgets.QFrame(self.menu_frame)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.input_personName = QtWidgets.QLineEdit(self.frame_5)
        self.input_personName.setMinimumSize(QtCore.QSize(360, 40))
        self.input_personName.setMaximumSize(QtCore.QSize(360, 40))
        self.input_personName.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.input_personName.setStyleSheet("QLineEdit:hover{\n"
"background:rgb(110,120,148);\n"
"border-radius: 15px;\n"
"}\n"
"\n"
"QLineEdit{\n"
"border: 2px solid rgb(26,169,253);\n"
"}")
        self.input_personName.setMaxLength(50)
        self.input_personName.setFrame(True)
        self.input_personName.setCursorPosition(0)
        self.input_personName.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.input_personName.setClearButtonEnabled(True)
        self.input_personName.setObjectName("input_personName")
        self.horizontalLayout_6.addWidget(self.input_personName)
        self.verticalLayout.addWidget(self.frame_5)
        self.frame_7 = QtWidgets.QFrame(self.menu_frame)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.input_imageName = QtWidgets.QLineEdit(self.frame_7)
        self.input_imageName.setMinimumSize(QtCore.QSize(310, 40))
        self.input_imageName.setMaximumSize(QtCore.QSize(310, 40))
        self.input_imageName.setStyleSheet("QLineEdit:hover{\n"
"background:rgb(110,120,148);\n"
"border-radius: 15px;\n"
"}\n"
"\n"
"QLineEdit{\n"
"border: 2px solid rgb(26,169,253);\n"
"}")
        self.input_imageName.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.input_imageName.setObjectName("input_imageName")
        self.horizontalLayout_7.addWidget(self.input_imageName)
        self.JPG = QtWidgets.QLabel(self.frame_7)
        self.JPG.setObjectName("JPG")
        self.horizontalLayout_7.addWidget(self.JPG)
        self.verticalLayout.addWidget(self.frame_7)
        self.frame_6 = QtWidgets.QFrame(self.menu_frame)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.bt_add = QtWidgets.QPushButton(self.frame_6)
        self.bt_add.clicked.connect(self.addToDataBase)
        self.bt_add.setMinimumSize(QtCore.QSize(360, 40))
        self.bt_add.setMaximumSize(QtCore.QSize(360, 40))
        self.bt_add.setStyleSheet("QPushButton:hover{\n"
"border: 5px solid rgb(246, 211, 45);\n"
"background:rgb(110,120,148);\n"
"border-radius: 15px;\n"
"}")
        self.bt_add.setObjectName("bt_add")
        self.horizontalLayout_4.addWidget(self.bt_add)
        self.verticalLayout.addWidget(self.frame_6)
        self.frame_8 = QtWidgets.QFrame(self.menu_frame)
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_8)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.bt_train = QtWidgets.QPushButton(self.frame_8)
        self.bt_train.clicked.connect(self.trainData)
        self.bt_train.setMinimumSize(QtCore.QSize(360, 40))
        self.bt_train.setMaximumSize(QtCore.QSize(360, 40))
        self.bt_train.setStyleSheet("QPushButton:hover{\n"
"border: 5px solid rgb(224, 27, 36);\n"
"background:rgb(110,120,148);\n"
"border-radius: 15px;\n"
"}")
        self.bt_train.setObjectName("bt_train")
        self.horizontalLayout_5.addWidget(self.bt_train)
        self.verticalLayout.addWidget(self.frame_8)
        self.horizontalLayout.addWidget(self.menu_frame)

        self.retranslateUi(databaseView)
        QtCore.QMetaObject.connectSlotsByName(databaseView)

    def retranslateUi(self, databaseView):
        _translate = QtCore.QCoreApplication.translate
        databaseView.setWindowTitle(_translate("databaseView", "DATABASE VIEW"))
        self.bt_load.setText(_translate("databaseView", "Load"))
        self.bt_search.setText(_translate("databaseView", "Search"))
        self.input_personName.setPlaceholderText(_translate("databaseView", "     Name of the person"))
        self.input_imageName.setPlaceholderText(_translate("databaseView", "     Set a image name"))
        self.JPG.setText(_translate("databaseView", "  .jpg"))
        self.bt_add.setText(_translate("databaseView", "Add"))
        self.bt_train.setText(_translate("databaseView", "Train"))

    def browseFile(self):
        self.img_display.clear()
        self.name_display.clear()
        self.fname = QFileDialog.getOpenFileName()
        pict = Image.open(self.fname[0])
        pict.save("tmp/new_person.jpg")
        self.img_display.setPixmap(QtGui.QPixmap(self.fname[0]))
        self.name_display.setText("Ready for searching ...")
 
    def encodeImage(self):
        from face import detector, nameFound
        detector(self.fname[0])
        self.img_display.clear()
        self.img_display.setPixmap(QtGui.QPixmap("tmp/face.jpg"))
        self.name_display.setText(nameFound())
        
    def addToDataBase(self):
        inputName = self.input_personName.text()
        imageName = self.input_imageName.text()
        if (inputName == "" and imageName == ""):
                self.input_personName.setStyleSheet("QLineEdit:hover{\n"
                        "background:rgb(110,120,148);\n"
                        "border-radius: 15px;\n"
                        "}\n"
                        "\n"
                        "QLineEdit{\n"
                        "border: 2px solid rgb(255,0,0);\n"
                        "}")
                self.input_imageName.setStyleSheet("QLineEdit:hover{\n"
                        "background:rgb(110,120,148);\n"
                        "border-radius: 15px;\n"
                        "}\n"
                        "\n"
                        "QLineEdit{\n"
                        "border: 2px solid rgb(255,0,0);\n"
                        "}")
                pass
        elif (inputName != "" and imageName == ""):
                self.input_imageName.setStyleSheet("QLineEdit:hover{\n"
                        "background:rgb(110,120,148);\n"
                        "border-radius: 15px;\n"
                        "}\n"
                        "\n"
                        "QLineEdit{\n"
                        "border: 2px solid rgb(255,0,0);\n"
                        "}")
                self.input_personName.setStyleSheet("QLineEdit:hover{\n"
                        "background:rgb(110,120,148);\n"
                        "border-radius: 15px;\n"
                        "}\n"
                        "\n"
                        "QLineEdit{\n"
                        "border: 2px solid rgb(26,169,253);\n"
                        "}")
                pass
        elif (inputName == "" and imageName != ""):
                self.input_personName.setStyleSheet("QLineEdit:hover{\n"
                        "background:rgb(110,120,148);\n"
                        "border-radius: 15px;\n"
                        "}\n"
                        "\n"
                        "QLineEdit{\n"
                        "border: 2px solid rgb(255,0,0);\n"
                        "}")
                self.input_imageName.setStyleSheet("QLineEdit:hover{\n"
                        "background:rgb(110,120,148);\n"
                        "border-radius: 15px;\n"
                        "}\n"
                        "\n"
                        "QLineEdit{\n"
                        "border: 2px solid rgb(26,169,253);\n"
                        "}")
                pass
        else:
            if not os.path.exists("DB/"+inputName.title()):
                os.makedirs("DB/"+inputName.title())
            if os.path.exists("DB/"+inputName.title()):
                picture = Image.open(r'tmp/new_person.jpg')
                picture.save("DB/"+inputName.title()+"/"+imageName+".jpg")
                self.input_personName.clear()
                self.input_personName.setStyleSheet("QLineEdit:hover{\n"
                        "background:rgb(110,120,148);\n"
                        "border-radius: 15px;\n"
                        "}\n"
                        "\n"
                        "QLineEdit{\n"
                        "border: 2px solid rgb(26,169,253);\n"
                        "}")
                self.input_imageName.clear()
                self.input_imageName.setStyleSheet("QLineEdit:hover{\n"
                        "background:rgb(110,120,148);\n"
                        "border-radius: 15px;\n"
                        "}\n"
                        "\n"
                        "QLineEdit{\n"
                        "border: 2px solid rgb(26,169,253);\n"
                        "}")

    def trainData(self):
        from face import encode
        if os.path.exists("DB/representations_vgg_face_MANUAL.pkl"):
            os.remove("DB/representations_vgg_face_MANUAL.pkl")
        encode("DB")    # Specify only the directory name



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    databaseView = QtWidgets.QWidget()
    ui = Ui_databaseView()
    ui.setupUi(databaseView)
    databaseView.show()
    sys.exit(app.exec_())
