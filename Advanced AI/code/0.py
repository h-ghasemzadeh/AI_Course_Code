import pandas as pd

df = pd.DataFrame({
    'column_name': [1, 2, 3, 4, 5]
})

df['new_column'] = df['column_name'].apply(lambda x: x * 2)
print(df)