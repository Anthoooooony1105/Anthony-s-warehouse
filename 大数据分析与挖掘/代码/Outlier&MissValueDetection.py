import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
ls=np.arange(228).reshape(4,57)
data = pd.DataFrame(ls)

edm=pd.read_csv("/Users/anthoooooony/Desktop/spambase_csv.csv")
edm.rename(columns={'class':'class1'},inplace=True)#改名，class作为python中类的名称会造成后续语法错误
edm.head()
edm.isnull()#缺失点检测


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.figure(figsize=(8,5))
sns.boxplot(data=edm.loc[:, ['word_freq_make', 'word_freq_address','word_freq_all','word_freq_3d',]])
# sns.boxplot(data=edm.loc[:, [0,1,2,4]])#方便老师运行，测试集上的绘图
# plt.savefig("C:/Users/86135/Desktop/test.png", dpi=300,format="png")#保存图片于桌面
plt.show()