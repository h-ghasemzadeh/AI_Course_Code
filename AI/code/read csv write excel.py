import pandas as pd
import scipy
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Boston_Housing.csv')

# Specify a writer
writer = pd.ExcelWriter('Boston_Housing.xlsx', engine='xlsxwriter')
# Write your DataFrame to a file     
df.to_excel(writer, 'boston')
# Save the result 
writer.close()