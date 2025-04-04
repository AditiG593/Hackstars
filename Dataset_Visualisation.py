
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

static_data = pd.read_csv("static_client_data.csv")
time_series_data = pd.read_csv("time_series_data.csv")
target_data = pd.read_csv("target_data.csv")
macro_scenarios = pd.read_csv("macro_scenarios.csv")

df = pd.read_csv("encoded.csv")



print(df.describe())

print(df.isnull().sum())

# Check the data types of each column
print(df.dtypes)



# Visualize 

static_data.select_dtypes(include=['float64', 'int64']).hist(bins=30, figsize=(15, 10), color='skyblue', edgecolor='black')
time_series_data.select_dtypes(include=['float64', 'int64']).hist(bins=30, figsize=(15, 10), color='skyblue', edgecolor='black')
target_data.select_dtypes(include=['float64', 'int64']).hist(bins=30, figsize=(15, 10), color='skyblue', edgecolor='black')
macro_scenarios.select_dtypes(include=['float64', 'int64']).hist(bins=30, figsize=(15, 10), color='skyblue', edgecolor='black')
df.select_dtypes(include=['float64', 'int64']).hist(bins=30, figsize=(15, 10), color='skyblue', edgecolor='black')


plt.suptitle('Distribution of Numeric Features')
plt.tight_layout()
plt.show()