import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('train.csv')
# Numerical columns
num_col = [col for col in df.columns if df[col].dtype != 'object']
print('Numerical columns :',num_col)


# Create subplots: 1 plot per column
fig, axs = plt.subplots(nrows=len(num_col))
# Ensure axs is iterable even if there is only one column
if len(num_col) == 1:
    axs = [axs]
# Plot each numerical column in a separate subplot
for i, column in enumerate(num_col):
    df.boxplot(column=column, ax=axs[i], vert=False)
    axs[i].set_title(column) 
plt.tight_layout() 
plt.show()

