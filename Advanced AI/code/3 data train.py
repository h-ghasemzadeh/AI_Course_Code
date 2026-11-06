import pandas as pd
import scipy
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt



# Load the dataset
df = pd.read_csv('train.csv')
print(df.head())


df.info()
print(df.duplicated())
print(df.describe())






# example for plot - age
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(df['age'], bins = 16, range = (df['age'].min(),df['age'].max()))
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Count of Passengers')
plt.show()


# Categorical columns
cat_col = [col for col in df.columns if df[col].dtype == 'object']
print('Categorical columns :',cat_col)
# Numerical columns
num_col = [col for col in df.columns if df[col].dtype != 'object']
print('Numerical columns :',num_col)

print(df[cat_col].nunique())



# Create subplots: 1 plot per column
fig, axs = plt.subplots(nrows=len(num_col), ncols=1, dpi=80, figsize=(10, 6))

# Ensure axs is iterable even if there is only one column
if len(num_col) == 1:
    axs = [axs]

# Plot each numerical column in a separate subplot
for i, column in enumerate(num_col):
    df.boxplot(column=column, ax=axs[i], vert=False)
    axs[i].set_title(column)  # Set the title as the column name

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# data cleaning
# Identify the quartiles 
i = 0
for col in num_col:
    q1, q3 = np.percentile(df[col], [25, 75])
# Calculate the interquartile range
    iqr = q3 - q1
# Calculate the lower and upper bounds
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
# Drop the outliers
    clean_data = df[(df[col] >= lower_bound) 
				& (df[col] <= upper_bound)]
    i+=1

print(clean_data.describe())



# Create subplots: 1 plot per column
fig, axs = plt.subplots(nrows=len(num_col), ncols=1, dpi=80, figsize=(10, 6))

# Ensure axs is iterable even if there is only one column
if len(num_col) == 1:
    axs = [axs]

# Plot each numerical column in a separate subplot
for i, column in enumerate(num_col):
    clean_data.boxplot(column=column, ax=axs[i], vert=False)
    axs[i].set_title(column)  # Set the title as the column name

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()





# write raw and clean data to excel file
# Specify a writer
writer = pd.ExcelWriter('train_clean_data.xlsx', engine='xlsxwriter')
# Write your DataFrame to a file     
df.to_excel(writer, 'raw data')
clean_data.to_excel(writer, 'clean data')
# Save the result 
writer.close()

# Determin unique data
print(df['embarked'].unique()) 

# Drop houses where the target is missing 
df.dropna(axis=0, subset=['embarked'], inplace=True) 
target = df.embarked 
print(target.unique()) 

# One hot encoding
one_hot_encoded_target = pd.get_dummies(target) 
print(one_hot_encoded_target)

one_hot_encoded_embarked = pd.get_dummies(df, columns = ['embarked']) 
print(one_hot_encoded_embarked)

#correlation
numerical_data = df.select_dtypes(include=[np.number])
print(numerical_data)
corr = numerical_data.corr()

plt.figure(dpi=130)
sns.heatmap(corr, annot=True, fmt= '.2f')
plt.show()
'''