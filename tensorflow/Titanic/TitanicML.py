# coding:utf-8

import pandas as pd  # 数据分析
import numpy as np  # 科学计算
from pandas import Series, DataFrame

data_train = pd.read_csv("train.csv")
print(data_train.head())

#################################################  [1] 数据探查  ########################################################

# 我们发现有一些列，比如说Cabin,有非常多的缺失值;另外一些我们感觉在此场景中会有影响的属性，比如Age，也有一些缺失值[”妇女小孩先走“]
data_train.info()

# [891 rows x 12 columns]
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):

# PassengerId    891 non-null int64     乘客ID
# Survived       891 non-null int64     是否获救
# Pclass         891 non-null int64     乘客等级
# Name           891 non-null object    乘客姓名
# Sex            891 non-null object    性别
# Age            714 non-null float64   年龄
# SibSp          891 non-null int64     堂兄弟/妹个数
# Parch          891 non-null int64     父母与小孩个数
# Ticket         891 non-null object    船票信息
# Fare           891 non-null float64   票价
# Cabin          204 non-null object    客仓
# Embarked       889 non-null object    登船港口

# dtypes: float64(2), int64(5), object(5)
# memory usage: 83.6+ KB

print(data_train.describe())

###################################### [2] 原始数据可视化（乘客各属性分布）  ################################################
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['BitstreamVeraSerif']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import matplotlib
matplotlib.matplotlib_fname()  # 将会获得matplotlib包所在文件夹

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
plt.title(u'获救情况（1为获救）')  # 标题
plt.ylabel(u'人数')

plt.show()


