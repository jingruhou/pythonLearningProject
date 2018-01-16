# coding:utf-8

# ,TV,radio,newspaper,sales
# 1,230.1,37.8,69.2,22.1
# 2,44.5,39.3,45.1,10.4
# 3,17.2,45.9,69.3,9.3
# 4,151.5,41.3,58.5,18.5
# 5,180.8,10.8,58.4,12.9

import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_csv('Advertising.csv')
# data.head()

if __name__ == "__main__":
    # 一、数据集信息：共4列 200行，每一行为一个特定的商品，前3列为输入特征，最后一列为输出特征

    # 输入特征：
    # （1）TV：该商品用于电视上的广告费用（以千元为单位，下同）
    # （2）radio：在广播媒体上投资的广告费用
    # （3）Newspaper：用于报纸媒体的广告费用

    # 输出特征：
    # Sales：该商品的销量
    path = 'Advertising.csv'
    # pandas读入
    data = pd.read_csv(path)
    x = data[['TV','radio','newspaper']]
    y = data['sales']

    # 二、收集、准备数据
    # plt.plot(data['TV'],y,'ro',label='TV')
    # plt.plot(data['radio'], y, 'g^',label='Radio')
    # plt.plot(data['newspaper'],y,'b*',label='Newspaper')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()

    plt.figure(figsize=(9,12))

    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')

    plt.grid()
    plt.subplot(312)
    plt.plot(data['radio'], y, 'g^')
    plt.title('Radio')

    plt.grid()
    plt.subplot(313)
    plt.plot(data['newspaper'], y, 'b*')
    plt.title('Newspaper')

    plt.grid()
    plt.tight_layout()
    plt.show()

    # 三、使用pandas来构建x[特征向量]和y[标签列]
    # features_cols =['TV','radio','newspaper']
    features_cols = ['TV', 'radio']
    X = data[features_cols]
    print X.head()

    print type(X)
    print X.shape # 返回表示DataFrame维度的元祖

    y = data['sales']
    print y.head()

    # 四、构建训练集与测试集
    from sklearn.cross_validation import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

    # 默认分割为75%的训练集，25%的测试集
    print X_train.shape # (150, 3)
    print y_train.shape # (150,)
    print X_test.shape  # (50, 3)
    print y_test.shape  # (50,)

    # 五、sklearn的线性回归模型
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    model = linreg.fit(X_train,y_train)

    print model # LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

    print linreg.intercept_ # 2.87696662232
    print linreg.coef_ # [ 0.04656457  0.17915812  0.00345046]
    # 由此得到各项系数：y=2.87696662232 + 0.04656457*TV  + 0.17915812*radio + 0.00345046*newspaper


    result = zip(features_cols,linreg.coef_)
    # [('TV', 0.046564567874150295), ('radio', 0.17915812245088839), ('newspaper', 0.0034504647111804343)]
    print result

    # 六、预测
    y_pred = linreg.predict(X_test)
    print y_pred
    print type(y_pred)

    # 七、模型评估

    # 对于分类问题，评价指标[evaluation metrics]是准确率，但这种方法不适用于回归问题。
    # 我们使用针对连续数值的评价指标[1]平均绝对误差-MAE[2]均方误差-MSE[3]均方根误差-RMSE

    print type(y_pred),type(y_test)
    print len(y_pred),len(y_test)
    print y_pred.shape,y_test.shape

    from sklearn import metrics
    import numpy as np

    sum_mean = 0
    for i in range(len(y_pred)):
         sum_mean+=(y_pred[i]-y_test.values[i])**2
         print "RMSE By hand:", np.sqrt(sum_mean/len(y_pred))


    # 八、可视化

    plt.figure()
    plt.plot(range(len(y_pred)),y_pred,'b',label='predict')
    plt.plot(range(len(y_pred)),y_test,'r',label='test')

    plt.legend(loc="upper right") # 显示图中的标签
    plt.xlabel("the number of sales")
    plt.ylabel("value of sales")
    plt.show()

    # 九、结果分析

    # 根据结果y=2.87696662232 + 0.04656457*TV  + 0.17915812*radio + 0.00345046*newspaper，发现newspaper的系数很小，
    # 进一步观察“收益-Newspaper”散点图，发现newspaper的线性关系并不明显，
    # 因此，尝试将这个特征移除，看看线性回归预测的结果的RMSE如何

    # 我们将newspaper这个特征移除之后，得到RMSE变小了，说明newspaper特征可能不适合作为预测销量的特征，
    # 于是，我们得到了新的模型。

    # 十、注意事项

    # 本模型虽然简单，但是它涵盖了机器学习的相当部分的内容。
    # 如使用75%的训练集和25%的测试集，这往往是探索机器学习模型的第一步。
    # 分析结果的权值和特征的数据分布，我们使用了最为简单的方法：直接删除;但这样做，任然得到了更好的预测结果。

    # 在机器学习中有“奥卡姆剃刀”的原理，即：如果能够用简单模型解决问题，则不使用更为复杂的模型。
    # 因为复杂模型往往增加了不确定性，造成过多的人力和物力成本，且容易过拟合。

