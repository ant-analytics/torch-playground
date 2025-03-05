import os
import pandas as pd
import numpy as np
import string
import random
from datetime import datetime
import time
import gc # garbage collector

# set random seed
rng = np.random.default_rng(12345)

# dataset parameters
num_samples = 1000
num_features = 50
folder_path = '/mnt/d/torch-playground/data'
extension = 'csv'
num_files = 5

# generate random string
def random_string(length=10, extension=extension):
    """Generate file name with a timestamp to ensure uniqueness"""
    letters = string.ascii_lowercase
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    return ''.join(rng.choice(list(letters), size=length)) + f'_{timestamp}.{extension}'

# generate dataset
def generate_dataset(num_samples=num_samples, num_features=num_features, num_files=num_files, folder_path=folder_path):
    """Generate multiple random datasets with given number of samples and features"""
    # create data folder if not exist
    os.makedirs(folder_path, exist_ok=True)

    # delete all files in the data folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

    # generate multiple datasets
    for i in range(num_files):
        # generate random data
        X = rng.normal(0, 1, (num_samples, num_features)) # mean = 0, std = 1
        y = rng.normal(0, 1, size=num_samples) # mean = 0, std = 1
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(num_features)])
        df['label'] = y
        output_path = os.path.join(folder_path, random_string())
        df.to_csv(output_path, index=False)
    print(f'Generated {num_files} datasets with {num_samples} samples and {num_features} features in {folder_path}')


# delete all created objects
del extension, folder_path, num_features, num_files, num_samples