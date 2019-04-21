import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)

residualtest = pd.read_csv('C:/Users/pyca/Desktop/毕业设计/4、多元线性回归/\
Multiple-Linear-Regression-master/Multiple-Linear-Regression-master/code/residualtest.csv')
print(residualtest['residual'].mean())

sns.residplot(x="number", y="residual", data=residualtest, scatter_kws={"s": 5})
plt.show()