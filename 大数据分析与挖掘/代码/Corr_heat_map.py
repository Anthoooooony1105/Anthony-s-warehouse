# coding: utf-8
import numpy as np
from random import randint, sample
import pandas as pd
#测试用数据，变量维度与实际数据匹配
#edm=pd.read_csv("C:/Users/86135/Desktop/spambase_csv.csv")
edm.rename(columns={'class':'class1'},inplace=True)#改名，class作为python中类的名称会造成后续语法错误
edm.head()
import matplotlib.pyplot as plt
#harvest = np.array(edm.iloc[:,0:56].corr())
harvest = np.array(edm.corr())#测试数据是线性相关的，所以热力图大面积都是黄色
plt.imshow(harvest)
plt.colorbar()
plt.tight_layout()#绘制相关系数热力图
#plt.savefig("C:/Users/86135/Desktop/test.png", dpi=300,format="png")#保存热力图
