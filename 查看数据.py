#!usr/bin/env python
#_*_ coding:utf-8 _*_
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl   #显示中文
def mul_lr():
    pd_data=pd.read_excel('C:\\Users\\pyca\\Desktop\\毕业设计\\job_info_num.xlsx')
    print('pd_data.head(10)=\n{}'.format(pd_data.head(10)))

# mul_lr()
pd_data=pd.read_excel('C:\\Users\\pyca\\Desktop\\毕业设计\\job_info_num.xlsx')
mpl.rcParams['font.sans-serif'] = ['SimHei']  #配置显示中文，否则乱码
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号，如果是plt画图，则将mlp换成plt
sns.pairplot(pd_data, x_vars=['工作经验','学历','公司规模','城市招聘发布数','行业招聘发布数'
                              ,'excel','python','sas','ppt','spss','hadoop','BI','mysql'], y_vars='薪资',kind="reg", size=5, aspect=0.7)
plt.show()#注意必须加上这一句，否则无法显示。