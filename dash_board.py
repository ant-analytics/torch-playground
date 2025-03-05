import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scripts.generate_dataset import *
from statsmodels.stats.descriptivestats import describe
plt.style.use('seaborn-v0_8-whitegrid') # set the style for plotting

# start measure time
start = time.time()

# generate multiple datasets
num_samples = 10000  
num_features = 50
folder_path = '/mnt/d/torch-playground/data'
extension = 'csv'
num_files = 5

generate_dataset(num_features=num_features, num_files=num_files, folder_path=folder_path, num_samples=num_samples)

# load the data
file_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# read each file and combine all into a single DataFrame
for file in file_list:
    df = pd.read_csv(file)
    if file == file_list[0]:
        combined_df = df
    else:
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# end measure time
end = time.time()
print(f'Execution time: {end - start} seconds')

# explore data
describe(combined_df, stats=['nobs', 'min', 'mean', 'max', 'percentiles'], percentiles=[5, 25, 75, 99]).T

# plot the distribution of the target variable
num_features = combined_df.shape[1] - 1  # assuming the last column is the target variable
num_cols = 5  # number of columns in the subplot grid
num_rows = (num_features + num_cols - 1) // num_cols  # calculate the number of rows needed

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
axes = axes.flatten()

for i in range(num_features):
    ax = axes[i]
    feature_name = combined_df.columns[i]
    ax.hist(combined_df[feature_name], bins=30, alpha=0.7)
    ax.set_title(f'Distribution of {feature_name}')

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
