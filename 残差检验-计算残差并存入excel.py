from sklearn.model_selection import train_test_split #这里是引用了交叉验证，分成训练集和测试集
from sklearn.linear_model import LinearRegression  #线性回归
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats


###读取数据
pd_data=pd.read_excel('C:\\Users\\pyca\\Desktop\\毕业设计\\job_info_num.xlsx')
X = pd_data.loc[:, ('工作经验', '公司规模', '城市招聘发布数', '行业招聘发布数'
                    , 'excel', 'python',  'hadoop', 'BI')]
y = pd_data.loc[:, '薪资']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)
print('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n,  y_test.shape={}'.format(X_train.shape, y_train.shape,
                                                                                          X_test.shape, y_test.shape))
linreg = LinearRegression()
model = linreg.fit(X_train, y_train)
feature_cols = ['工作经验', '学历', '公司规模', '城市招聘发布数', '行业招聘发布数'
    , 'excel', 'python', 'hadoop', 'BI', '薪资']
B = list(zip(feature_cols, linreg.coef_))
y_pred = linreg.predict(X_test)



L = []
for i in range(len(y_pred)):
    a = y_pred[i] - y_test.values[i]
    L.append(a)
print(len(L))

# name=['residual']
# test=pd.DataFrame(columns=name,data=L)#数据有一列列，列名为residual
# test.to_csv('C:/Users/pyca/Desktop/毕业设计/4、多元线性回归/Multiple-Linear-Regression-master/Multiple-Linear-Regression-master/code/\
# residualtest.csv',encoding='gbk')
