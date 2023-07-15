# coding: utf-8
import numpy as np
from random import randint, sample
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


#edm=pd.read_csv("C:/Users/86135/Desktop/spambase_csv.csv")
edm.rename(columns={'class':'class1'},inplace=True)#改名，class作为python中类的名称会造成后续语法错误
edm.head()
data_many=edm[(edm['class1']!= 1)&(edm['class1']!= 0)]
print(data_many)#筛选类不为0或者1的数据
data_negative=[edm<0]#筛选数据小于0的情况
print(data_negative)
