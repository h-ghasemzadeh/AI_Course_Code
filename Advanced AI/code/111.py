# importing libraries
import pandas as pd




# Load the dataset
df = pd.read_csv('diabetes.csv')

df.info()
df_renamed = df.rename(columns={'Pregnancies': 'Patient'})



df['BloodPressure'] = df['BloodPressure'].astype('float')


df.to_csv('diabetes_pro.csv', index=False)

