# coding:utf-8

# 4.3 数据预处理

# sklearn.preprocessing软件包提供了集中常用的效用函数和变换器类，用于将原始特征向量更改为更适合下游估计器的表示形式。
# 一般而言 ，学习算法会从数据集的标准化中受益。如果在该组中存在一些异常值，则鲁棒的比例调整器或变压器更合适。
# 在比较不同缩放器对数据与异常值的影响时，不同缩放器、变换器和归一化在包含边际离群值的数据集上的行为将突出显示。

# 4.3.1 标准化或均值去除和方差缩放
from sklearn import preprocessing
import numpy as np

X_train = np.array([[1., -1., 2.],
                   [2., 0., 0.],
                   [0., 1., -1.]])

X_scaled = preprocessing.scale(X_train)
print '调用preprocessing.scale()函数标准化数据： ','\n',  X_scaled

X_mean = X_scaled.mean(axis=0)
X_std = X_scaled.std(axis=0)
X1 = (X_train - X_mean) / X_std
print '通过标准化公式计算得到: ', '\n', X1

# StandardScaler()方法也可以对数据进行标准化处理
scaler = preprocessing.StandardScaler()
X_scaled1 = scaler.fit_transform(X_train)
print X_scaled1, '\n'

# 将特征的取值缩小至0-1之间，采用MinMaxScaler函数
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X_train)
print X_minmax, '\n'

# 正则化 

