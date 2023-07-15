from sklearn.linear_model import LogisticRegression#y有关逻辑回归的包
from sklearn.svm import SVC
import os
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
import sys
from PyQt5.QtGui import QPixmap, QCursor
from PyQt5.QtWidgets import QAbstractItemView, QGraphicsPixmapItem, QGraphicsScene, QTableWidgetItem, QApplication, \
    QMainWindow, QWidget, QLabel, QTableWidget, QPushButton, QGraphicsView, QLineEdit, QMenuBar, QStatusBar, \
    QRadioButton
import wordcloud
import numpy as np
from random import randint,sample
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb #载入xgboost算法的包
from sklearn.metrics import accuracy_score#计算分类正确率的包
from PyQt5 import QtCore, QtGui, QtWidgets


class read_data_csv:
    def data(self):
        # 判断各个数据之间是否存在线性关系   一次只能画1张
        data_all = pd.read_csv("/Users/anthoooooony/Desktop/spambase_csv.csv", header=None)
        data_all = np.array(data_all).reshape(4602, 58)
        for i in range(1, 4602):
            for j in range(0, 58):
                data_all[i, j] = float(data_all[i, j])
                if (j == 57):
                    data_all[i, j] = int(data_all[i, j])
        # print(data_all)
        return data_all


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1362, 778)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(730, 630, 159, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(730, 680, 159, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(890, 630, 141, 31))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(890, 680, 141, 31))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(1040, 630, 141, 32))
        self.pushButton_5.setObjectName("pushButton_5")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(620, 60, 641, 551))
        self.graphicsView.setObjectName("graphicsView")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(50, 470, 531, 151))
        self.textBrowser.setObjectName("textBrowser")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 330, 151, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(530, 20, 300, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(50, 140, 141, 16))
        self.label_3.setObjectName("label_3")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(270, 260, 141, 181))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.radioButton = QtWidgets.QRadioButton(self.layoutWidget)
        self.radioButton.setObjectName("radioButton")
        self.verticalLayout_2.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.layoutWidget)
        self.radioButton_2.setObjectName("radioButton_2")
        self.verticalLayout_2.addWidget(self.radioButton_2)
        self.radioButton_3 = QtWidgets.QRadioButton(self.layoutWidget)
        self.radioButton_3.setObjectName("radioButton_3")
        self.verticalLayout_2.addWidget(self.radioButton_3)
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(1040, 680, 141, 31))
        self.pushButton_6.setObjectName("pushButton_6")
        self.commandLinkButton = QtWidgets.QCommandLinkButton(self.centralwidget)
        self.commandLinkButton.setGeometry(QtCore.QRect(410, 130, 193, 41))
        self.commandLinkButton.setObjectName("commandLinkButton")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(160, 190, 411, 16))
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1362, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "画标准差柱状图"))
        self.pushButton_2.setText(_translate("MainWindow", "画均值条形图"))
        self.pushButton_3.setText(_translate("MainWindow", "画相关系数图"))
        self.pushButton_4.setText(_translate("MainWindow", "画各个成分的饼图"))
        self.pushButton_5.setText(_translate("MainWindow", "画部分成分的箱线图"))
        self.label.setText(_translate("MainWindow", "请选择训练的模型："))
        self.label_2.setText(_translate("MainWindow", "welcome to machine learning displayer"))
        self.label_3.setText(_translate("MainWindow", "请选择使用的数据集："))
        self.radioButton.setText(_translate("MainWindow", "LogisticRegression"))
        self.radioButton_2.setText(_translate("MainWindow", "SVM"))
        self.radioButton_3.setText(_translate("MainWindow", "XGBoost"))
        self.pushButton_6.setText(_translate("MainWindow", "词云可视化"))
        self.commandLinkButton.setText(_translate("MainWindow", "Click_to_choose_a_file"))




        #开始设置链接:
        self.pushButton.clicked.connect(self.DrawPicture1)
        self.pushButton_3.clicked.connect(self.DrawPicture3)
        self.pushButton_2.clicked.connect(self.DrawPicture2)
        self.pushButton_4.clicked.connect(self.DrawPicture4)
        self.pushButton_5.clicked.connect(self.DrawPicture5)
        self.pushButton_6.clicked.connect(self.DrawPicture6)
        self.commandLinkButton.clicked.connect(self.ChooseFile)

        # 开始弄左边：
        self.commandLinkButton.clicked.connect(self.ChooseFile)
        self.radioButton.clicked.connect(self.ML_LogisticRegression)
        self.radioButton_2.clicked.connect(self.ML_SVM)
        self.radioButton_3.clicked.connect(self.ML_XGBoost)


    def DrawPicture1(self):
        data_all = read_data_csv.data(self)
        column_std = np.zeros(54)
        x = np.ones(54)
        for i in range(1, 55):
            x[i - 1] = i
        for i in range(0, 54):
            i_column_data = np.array(data_all[1:len(data_all), i]).reshape(1, len(data_all) - 1)
            i_column_mean = np.mean(i_column_data)
            i_column_std = np.std(i_column_data)
            column_std[i] = i_column_std
        plt.bar(x, column_std)
        #构造文件名称
        figure_save_path = "/Users/anthoooooony/Desktop/data"
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(figure_save_path, '1' + 'exam.png'))  # 第一个是指存储路径，第二个是图片名字
        path_std = str(figure_save_path) + '/' +'1'+'exam.png'
        #开始GUI显示
        file = path_std
        pix = QPixmap(file)
        pix = pix.scaled(650, 580)
        item = QGraphicsPixmapItem(pix)
        item.setScale(1)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.graphicsView.setScene(scene)
        self.graphicsView.repaint()
        return

    def DrawPicture2(self):
        # 平均值做折线图，可以得到词或者字符出现最多的情况

        plt.close()
        self.graphicsView.repaint()
        data_all = read_data_csv.data(self)
        title = data_all[0, :]
        x = np.ones(54)
        data_mean = np.zeros(54)
        for i in range(1, 55):
            x[i - 1] = i
            data_mean[i - 1] = np.mean(data_all[1:len(data_all), i - 1])
        # print(data_mean)
        plt.bar(x, data_mean)
        # plt.xticks(x, title[0:54], rotation=60)
        # 构造文件名称
        figure_save_path = "/Users/anthoooooony/Desktop/data"
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(figure_save_path, '2' + 'exam.png'))  # 第一个是指存储路径，第二个是图片名字
        path_std = str(figure_save_path) + '/' + '2' + 'exam.png'
        #开始GUI画图
        self.graphicsView.setScene(QGraphicsScene())
        file = path_std
        pix = QPixmap(file)
        pix = pix.scaled(650, 580)
        item = QGraphicsPixmapItem(pix)
        item.setScale(1)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.graphicsView.setScene(scene)
        self.graphicsView.repaint()
        return

    def DrawPicture3(self):
        # 画相关系数图
        plt.close()
        data_all = read_data_csv.data(self)
        for i in range(0, 1):
            fig = plt.figure()
            for j in range(0, 54):
                ax = fig.add_subplot(6, 9, j + 1)
                plt.plot(data_all[1:len(data_all), i], data_all[1:len(data_all), j])
            #     print(j)
            # plt.show()
        # 构造文件名称
        figure_save_path = "/Users/anthoooooony/Desktop/data"
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(figure_save_path, '3' + 'exam.png'))  # 第一个是指存储路径，第二个是图片名字
        path_std = str(figure_save_path) + '/' + '3' + 'exam.png'
        #开始GUI绘图
        self.graphicsView.setScene(QGraphicsScene())
        file = path_std
        pix = QPixmap(file)
        pix = pix.scaled(650, 580)
        item = QGraphicsPixmapItem(pix)
        item.setScale(1)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.graphicsView.setScene(scene)
        self.graphicsView.repaint()
        return

    def DrawPicture4(self):
        # 画饼图
        plt.close()
        data_all = read_data_csv.data(self)
        title = data_all[0, :]
        x = np.ones(54)
        data_mean = np.zeros(54)
        for i in range(1, 55):
            x[i - 1] = i
            data_mean[i - 1] = np.mean(data_all[1:len(data_all), i - 1])
        # print(data_mean)
        plt.pie(data_mean, labels=title[0:54])
        # 构造文件名称
        figure_save_path = "/Users/anthoooooony/Desktop/data"
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(figure_save_path, '4' + 'exam.png'))  # 第一个是指存储路径，第二个是图片名字
        path_std = str(figure_save_path) + '/' + '4' + 'exam.png'
        # 开始GUI绘图
        file = path_std
        pix = QPixmap(file)
        pix = pix.scaled(650, 580)
        item = QGraphicsPixmapItem(pix)
        item.setScale(1)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.graphicsView.setScene(scene)
        self.graphicsView.repaint()
        return

    def DrawPicture5(self):
        # 画箱线图
        plt.close()
        data_all = read_data_csv.data(self)
        for i in range(0, 1):
            plt.boxplot(data_all[1:len(data_all), i])
            # plt.show()
            # 构造文件名称
            figure_save_path = "/Users/anthoooooony/Desktop/data"
            if not os.path.exists(figure_save_path):
                os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
            plt.savefig(os.path.join(figure_save_path, '5' + 'exam.png'))  # 第一个是指存储路径，第二个是图片名字
            path_std = str(figure_save_path) + '/' + '5' + 'exam.png'
            # 开始GUI绘图
            file = path_std
            pix = QPixmap(file)
            pix = pix.scaled(650, 580)
            item = QGraphicsPixmapItem(pix)
            item.setScale(1)
            scene = QGraphicsScene()  # 创建场景
            scene.addItem(item)
            self.graphicsView.setScene(scene)
            self.graphicsView.repaint()
        return

    def DrawPicture6(self):
        text = open('/Users/anthoooooony/Desktop/data的副本.csv', encoding="utf-8").read()
        text = text.replace("“ ", "").replace("。", "").replace(",", "").replace("!", "").replace("?", "").replace("‘",
                                                                                                                  "").replace(
            ".", "").replace("’", "")  # 去掉杂质，提纯

        wc = wordcloud.WordCloud().generate(text)
        # 显示词云
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")  # 关闭坐标轴
        # 构造文件名称
        figure_save_path = "/Users/anthoooooony/Desktop/data"
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(figure_save_path, '6' + 'exam.png'))  # 第一个是指存储路径，第二个是图片名字
        path_std = str(figure_save_path) + '/' + '6' + 'exam.png'
        # 开始GUI绘图
        file = path_std
        pix = QPixmap(file)
        pix = pix.scaled(650, 580)
        item = QGraphicsPixmapItem(pix)
        item.setScale(1)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.graphicsView.setScene(scene)
        self.graphicsView.repaint()
        return

    def ChooseFile(self):
        self.label_4.setText("/Users/anthoooooony/Desktop/spambase_csv.csv")
        return

    def ML_LogisticRegression(self):
        data_all = read_data_csv.data(self)
        X = data_all[1:4601, 0:56]
        Y = data_all[1:4601, 57]
        Y = Y.astype('int')
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=1)  # 划分测试集训练集
        classmodel = LogisticRegression()
        classmodel.fit(Xtrain, Ytrain)
        ypred = classmodel.predict(Xtrain)
        train_accuracy = accuracy_score(Ytrain, ypred)
        # print("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
        ytestpred = classmodel.predict(Xtest)
        test_accuracy = accuracy_score(Ytest, ytestpred)
        self.textBrowser.setText("\n\nTrain Accuary: " +str(train_accuracy * 100.0) + '\n' + "\nTest Accuary:" + str(test_accuracy * 100.0))
        self.textBrowser.repaint()
        return

    def ML_SVM(self):
        data_all = read_data_csv.data(self)
        X = data_all[1:4601, 0:56]
        X = np.delete(X, [28, 30, 31, 32, 34], axis=1)
        Y = data_all[1:4601, 57]
        Y = Y.astype('int')
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=1)  # 划分测试集训练集
        # 线性核为：kernel="linear"；二次多项式核为：kernel="poly", degree=2；三次多项式核为：kernel="poly", degree=3；径向核为：kernel="rbf"；S核为：kernel="sigmoid"
        classmodel = SVC(kernel="linear", degree=2, random_state=123)
        classmodel.fit(Xtrain, Ytrain)
        ypred = classmodel.predict(Xtrain)
        train_accuracy = accuracy_score(Ytrain, ypred)
        # print("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
        ytestpred = classmodel.predict(Xtest)
        test_accuracy = accuracy_score(Ytest, ytestpred)
        self.textBrowser.setText(
            "\n\nTrain Accuary:" + str(train_accuracy * 100.0) + '\n' + "\nTest Accuary: " + str(
                test_accuracy * 100.0))
        self.textBrowser.repaint()
        return

    def ML_XGBoost(self):
        data_all = read_data_csv.data(self)
        X = data_all[1:4601, 0:56]
        Y = data_all[1:4601, 57]
        Y = Y.astype('int')
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=1)  # 划分测试集训练集
        D_train = xgb.DMatrix(Xtrain, Ytrain)
        D_test = xgb.DMatrix(Xtest, Ytest)
        param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
        bst = xgb.train(param, D_train, 5)
        train_preds = bst.predict(D_train)
        train_predictions = [round(value) for value in train_preds]
        y_train = D_train.get_label()
        train_accuracy = accuracy_score(Ytrain, train_predictions)
        # print("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
        preds = bst.predict(D_test)
        predictions = [round(value) for value in preds]
        y_test = D_test.get_label()
        test_accuracy = accuracy_score(Ytest, predictions)
        # print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
        self.textBrowser.setText(
            "\n\nTrain Accuary:" + str(train_accuracy * 100.0) + '\n' + "\nTest Accuary: " + str(
                test_accuracy * 100.0))
        self.textBrowser.repaint()
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

