import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

df = pd.read_csv('Depth-temperature.csv')
df_binary = df[['Depth (m)', 'Temperature ( c )']]

# Taking only the selected two attributes from the dataset
df_binary.columns = ['Depth (m)', 'Temperature ( c )']
#display the first 5 rows
df_binary.head()

#plotting the Scatter plot to check relationship between Depth and Temperature
sns.lmplot(x ="Depth (m)", y ="Temperature ( c )", data = df_binary, order = 2, ci = None)
plt.show()

b = estimate_coef(x, y)
print("Estimated coefficients:\nb_0 = {} \\nb_1 = {}".format(b[0], b[1]))