# coding: utf-8
import numpy as np
from random import randint,sample
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import warnings
warnings.filterwarnings("ignore")
Traindata = {
  'Linear': [93.70,93.70,93.66,93.66,93.63,93.51],
  'Two': [67.14,67.14,67.14,67.14,67.14,67.14],
  'Three': [65.22,63.39,63.39,63.39,63.39,63.39],
  'Gauss': [74.22,74.22,74.22,74.22,74.22,74.22],
  'S': [69.60,69.60,69.60,69.60,69.60,69.60],
}
x=[0,1,2,3,4,5]
y1 = Traindata['Linear']
y2 = Traindata['Two']
y3 = Traindata['Three']
y4 = Traindata['Gauss']
y5 = Traindata['S']

plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.xlabel("依次删去的变量个数") #X轴标签
plt.ylabel("准确率")
plt.title("SVM方差过滤分类准确率比较图(训练集)") #标题
plt.grid()
plt.plot(x, y1, marker='v', mec='r', mfc='w')
plt.plot(x, y2, marker='8', ms=10)
plt.plot(x, y3, marker='s', mec='r', mfc='w')
plt.plot(x, y4, marker='X', ms=10)
plt.plot(x, y5, marker='d', mec='r', mfc='w')
plt.legend(["线性核", "二次项核","三次项核", "高斯核","S核"])  # 让图例生效
plt.savefig('C:/Users/86135/Desktop/SVNTRAIN.png', dpi=300)
plt.show()

Testdata = {
  'Linear': [91.81,91.81,91.81,91.88,92.03,92.03],
  'Two': [65.80,65.80,65.80,65.80,65.80,65.80],
  'Three': [65.22,63.41,63.41,63.41,63.41,63.41],
  'Gauss': [72.32,72.32,72.32,72.32,72.32,72.32],
  'S': [71.38,71.38,71.38,71.38,71.38,71.38],
}
x=[0,1,2,3,4,5]
y11 = Testdata['Linear']
y22 = Testdata['Two']
y33 = Testdata['Three']
y44 = Testdata['Gauss']
y55 = Testdata['S']

plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.xlabel("依次删去的变量个数") #X轴标签
plt.ylabel("准确率")
plt.title("SVM方差过滤分类准确率比较图(测试集)") #标题
plt.grid()
plt.plot(x, y11, marker='v', mec='r', mfc='w')
plt.plot(x, y22, marker='8', ms=10)
plt.plot(x, y33, marker='s', mec='r', mfc='w')
plt.plot(x, y44, marker='X', ms=10)
plt.plot(x, y55, marker='d', mec='r', mfc='w')
plt.legend(["线性核", "二次项核","三次项核", "高斯核","S核"])  # 让图例生效
plt.savefig('C:/Users/86135/Desktop/SVMtest.png', dpi=300)
plt.show()
