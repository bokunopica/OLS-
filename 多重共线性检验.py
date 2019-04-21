import pandas as pd
import scipy.stats as stats




pd_data=pd.read_excel('C:\\Users\\Administrator\\Desktop\\毕业设计\\job_info_num.xlsx')
X = pd_data.loc[:, ('工作经验', '学历', '公司规模', '城市招聘发布数', '行业招聘发布数'
                    , 'excel', 'python', 'sas', 'ppt', 'spss', 'hadoop', 'BI', 'mysql')]
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





mul_test(11,12)
mul_test(11,13)
mul_test(12,13)