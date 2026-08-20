from datasets import load_dataset
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

ds = load_dataset("b-mc2/sql-create-context", split = 'train')

rename_mapping = {
    'question':'instructions',
    'answer' : 'response'
}

for old_name , new_name in rename_mapping.items():
    ds = ds.rename_column(old_name, new_name)

df = ds.to_pandas()
print(df.columns)
print(len(df))
def combine_columns(row):
    return f'Instructions: {row["instructions"]}\nContext: {row["context"]}\nResponse: {row["response"]}'

df['combined_column'] = df.apply(combine_columns, axis=1)
df['combined_column_len'] = df['combined_column'].astype(str).apply(len)
print('*' * 20)
print(df['combined_column_len'].describe())
print('*' * 20)
print(df[['combined_column','combined_column_len']].sort_values(by='combined_column_len', ascending=False).head())

percentiles = df['combined_column_len'].quantile([0.90 , 0.95])
print(percentiles)

filtered_df = df[df['combined_column_len'] <= 360]

print(f'lenght of the filtered dataset: {len(filtered_df)}')
print(f'length of the original dataset: {len(df)}')
print(f'rows dropped: {len(df) - len(filtered_df)}')
print(f"maximum char lenght in the new dataset is : {filtered_df['combined_column_len'].max()}")
filtered_df = filtered_df.reset_index(drop=True)
print(filtered_df.head())

random_sample = filtered_df.sample(n=4000,random_state=42)
random_sample = random_sample.reset_index(drop=True)
print(random_sample.head())
print(len(random_sample))

train_data , val_data = train_test_split(random_sample,test_size=0.1,random_state=42)
print(len(train_data))
print(len(val_data))

train_data_df = train_data.to_csv('train_data.csv', index=False, )
val_data_df = val_data.to_csv('val_data.csv', index=False)
