#!/usr/bin/python
# -*- coding: <encoding name> -*-
from sklearn.model_selection import train_test_split #这里是引用了交叉验证，分成训练集和测试集
from sklearn.linear_model import LinearRegression  #线性回归
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats

###读取数据
pd_data=pd.read_excel('C:\\Users\\Administrator\\Desktop\\毕业设计\\job_info_num.xlsx')
X = pd_data.loc[:, ('工作经验', '公司规模', '城市招聘发布数', '行业招聘发布数'
                    , 'excel', 'python',  'hadoop', 'BI')]
y = pd_data.loc[:, '薪资']
Xlist = ['工作经验', '学历', '公司规模', '城市招聘发布数', '行业招聘发布数']
Dlist = [ 'excel', 'python', 'sas', 'ppt', 'spss', 'hadoop', 'BI', 'mysql']
Mul_list = ['工作经验', '学历', '公司规模', '城市招聘发布数', '行业招聘发布数'
                    , 'excel', 'python', 'sas', 'ppt', 'spss', 'hadoop', 'BI', 'mysql']


def mul_test(a,b):
    array_values = pd_data.values
    x1 = [i[a-1] for i in array_values]
    x2 = [i[b-1] for i in array_values]

    print("use Pearson,parametric tests",Mul_list[a-1],"and",Mul_list[b-1])
    r, p = stats.pearsonr(x1, x2)
    print("pearson r**2:", r ** 2)
    vif = (1/(1-r ** 2))
    print("方差膨胀因子 Vif",vif)
    print("pearson p:", p)
    print('-----------------------------------------------------')



##OLS回归模型检验
def model_test(X,y):
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())
##主要的检验结果例如：
# 1、线性关系显著性检验
# H0：β=0，H1：β≠0，检验方法：t检验，显著性水平假设为0.01
# 结果中变量的P>|t| 均小于0.01，原假设不成立，变量与应变量间关系显著
# 2、检验多元回归模型
# H0:β1=β2=β3=β4=β5...=0,H1:at least one βk≠0，检验方法：F检验，显著性水平假设为0.01
# 结果中Prob (F-statistic):           8.25e-50 小于0.01，原假设不成立，多元回归模型成立
# 3、模型拟合度检验
# R-squared:                       0.869
# Adj. R-squared:                  0.861   根据自变量数量修正后的R方，自变量数量越多，修正后R方越小
# R-squared越大，模型拟合越好。


##变量显著性检验;n是指第几项普通变量
def R2_test_normal(n):
    print(Xlist[n-1])
    array_values = pd_data.values
    a = [i[n-1] for i in array_values]
    y = [i[16] for i in array_values]
    print("use Pearson,parametric tests","X",n,"and y")
    r, p = stats.pearsonr(a, y)
    print("pearson r**2:", r ** 2)
    print("pearson p:", p)
    if p < 0.05:
        print('P检验通过')
    else:
        print('P检验不通过')
    print('-----------------------------------------------------')


##虚拟变量显著性检验;n是指第几项虚拟变量
def R2_test_virtual(n):
    print(Dlist[n-1])
    array_values = pd_data.values
    a = [i[n+4] for i in array_values]
    y = [i[16] for i in array_values]
    print("use Pearson,parametric tests","D",n,"and y")
    r, p = stats.pearsonr(a, y)
    print("pearson r**2:", r ** 2)
    print("pearson p:", p)
    if p < 0.05:
        print('P检验通过')
    else:
        print('P检验不通过')
    print('-----------------------------------------------------')




def mul_lr():   #续前面代码
    #剔除日期数据，一般没有这列可不执行，选取以下数据http://blog.csdn.net/chixujohnny/article/details/51095817
    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.005,random_state=5)
    print ('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n,  y_test.shape={}'.format(X_train.shape,y_train.shape, X_test.shape,y_test.shape))
    linreg = LinearRegression()
    model=linreg.fit(X_train, y_train)
    print (model)
    # 训练后模型截距
    print (linreg.intercept_)
    # 训练后模型权重（特征个数无变化）
    print (linreg.coef_)
    feature_cols = ['工作经验','学历','公司规模','城市招聘发布数','行业招聘发布数'
                              ,'excel','python','hadoop','BI','薪资']
    B = list(zip(feature_cols, linreg.coef_))
    print(B)
    print('输出R方为', linreg.score(X, y))
    y_pred = linreg.predict(X_test)
    print(y_pred)  # n个变量的预测结果
    sum_mean=0
    for i in range(len(y_pred)):
        sum_mean+=(y_pred[i]-y_test.values[i])**2
    sum_erro=np.sqrt(sum_mean/40)  #这个10是你测试级的数量
    # calculate RMSE by hand 均方根误差
    print ("RMSE by hand（均方根误差）:",sum_erro)
    #做ROC曲线
    plt.figure()
    plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
    plt.plot(range(len(y_pred)),y_test,'r',label="test")
    plt.legend(loc="upper right") #显示图中的标签
    plt.xlabel("numbers")
    plt.ylabel('salary')
    plt.show()


def main():
    #多重共线性检验
    # mul_test()
    #模型检验
    # model_test(X,y)
    # #各标准变量R2检验
    # for i in range(1,6): ##X1-X5
    #     R2_test_normal(i)
    # #各虚拟变量R2检验
    # for i in range(1,9): ##D1-D8
    #     R2_test_virtual(i)
    # 回归
    mul_lr()
main()