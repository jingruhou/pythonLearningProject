# coding:utf-8

# 5.1,3.5,1.4,0.2,Iris-setosa
# 4.9,3.0,1.4,0.2,Iris-setosa
# 4.7,3.2,1.3,0.2,Iris-setosa
# 4.6,3.1,1.5,0.2,Iris-setosa
# 5.0,3.6,1.4,0.2,Iris-setosa

# 数据描述

# 该数据集共包括150行，每行1个样本。每个样本有5个字段，
# 分别是：花鄂长度（cm）、花鄂宽度（cm）、花瓣长度（cm）、花瓣宽度（cm）、类别（Iris Setosa、Iris Versicolour、Iris Virginica）

# 数据集特征：多变量 记录数：150 属性特征：实数 属性数目：4 相关应用：分类

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":

    # 一、数据加载
    path = 'iris.data'

    # 路径，浮点型数据，逗号分割，第四列使用函数iris_type单独处理
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})

    # 将数据的0到3列组成x，第4列得到y
    x, y = np.split(data, (4,), axis=1)
    # 为了可视化，仅使用前两列特征
    x = x[:, :2]
    # print x
    # print y

    # 二、模型构建
    logreg = LogisticRegression()  # Logistic 回归模型
    logreg.fit(x, y.ravel())  # 根据数据[x ,y]，计算回归参数

    # 三、画图
    N, M = 500, 500  # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围

    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)

    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    y_hat = logreg.predict(x_test)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入形状相同

    plt.pcolormesh(x1, x2, y_hat, cmap=plt.cm.prism)  # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.prism)  # 样本的显示

    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.show()

    # 训练集上的预测结果
    y_hat = logreg.predict(x)
    y = y.reshape(-1)
    print y_hat.shape
    print y.shape

    result = y_hat == y

    print y_hat
    print y
    print result

    c = np.count_nonzero(result)
    print c
    print 'Accuracy: %.2f%%' % (100 * float(c) / float(len(result)))
