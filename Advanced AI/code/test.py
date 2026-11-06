import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv ('train.csv')

# Numerical columns
num_col = [col for col in df.columns if df[col].dtype != 'object']
print('Numerical columns :',num_col)


fig, axs = plt.subplots(nrows=len(num_col), dpi=80, figsize=(10, 6))
# Ensure axs is iterable even if there is only one column
if len(num_col) == 1:
 axs = [axs]
# Plot each numerical column in a separate subplot
for i, column in enumerate(num_col):
 df.boxplot(column=column, ax=axs[i], vert=False)
 axs[i].set_title(column) 
# Set the title as the column name
plt.tight_layout() 
# Adjust layout to prevent overlap
plt.show()

i = 0
for col in num_col:
    q1, q3 = np.percentile(df[col], [25, 75])
# Calculate the interquartile range
    iqr = q3 - q1
# Calculate the lower and upper bounds
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
# Drop the outliers
clean_data = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
i+=1


print(df.describe())
print(clean_data.describe())

numerical_data = df.select_dtypes(include=[np.number])
print(numerical_data)
corr = numerical_data.corr()
plt.figure(dpi=130)
sns.heatmap(corr, annot=True, fmt= '.2f')
plt.show()
