import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy.stats
import matplotlib.pyplot as plt
import pylab

data = pd.read_csv('residualtest.csv')
x = data['residual']

#Shapiro-Wilk Test: https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
shapiro_test, shapiro_p = scipy.stats.shapiro(x)
print("Shapiro-Wilk Stat:",shapiro_test, " Shapiro-Wilk p-Value:", shapiro_p)

k2, p = scipy.stats.normaltest(x)
print('p:',p)


#Another method to determining normality is through Quantile-Quantile Plots.
scipy.stats.probplot(x, dist="norm", plot=pylab)
pylab.show()


def ecdf(data):
    #Compute ECDF
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

# Compute empirical mean and standard deviation

# Number of samples
n = len(data['residual'])

# Sample mean
mu = np.mean(data['residual'])

# Sample standard deviation
std = np.std(data['residual'])

print('Mean residual: ', mu, 'with standard deviation of +/-', std)

#Random sampling of the data based off of the mean of the data.
normalized_sample = np.random.normal(mu, std, size=10000)
x_temperature, y_temperature = ecdf(data['residual'])
normalized_x, normalized_y = ecdf(normalized_sample)

# Plot the ECDFs
fig = plt.figure(figsize=(8, 5))
plt.plot(normalized_x, normalized_y)
plt.plot(x_temperature, y_temperature, marker='.', linestyle='none')
plt.ylabel('ECDF')
plt.xlabel('Residual')
plt.legend(('Normal Distribution', 'Sample data'))
plt.show()