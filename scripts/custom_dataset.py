import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# start time measurement
start = time.time()

# with open(file_path, 'r') as f:
#     header = f.readline().strip().split(',')
#     features = header[:-1]
#     target = header[-1]

# print(f'this is a list of features: {features}')
# print(f'this is the target: {target}')

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        """
        Args:
            folder (string): Path to the folder containing the dataset
            read and load all csv files from the folder
        """
        self.folder_path = folder_path
        self.file_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
        self.data = []
        for file in self.file_list:
            data = np.loadtxt(file, delimiter=',', skiprows=1) # skip first row as it contains the header
            self.data.append(data)
        self.data = np.vstack(self.data) # combine all data into one numpy array
        self.X = torch.from_numpy(self.data[:, :-1]).float() # select all rows and all columns except the last one, feature matrix
        self.y = torch.from_numpy(self.data[:, -1]).float() # select all rows and the last column, target vector

    def __len__(self):
        """
        return the length of the dataset
        """
        return len(self.X)
    def __getitem__(self, idx):
        """
        return tuple of features and target at the given index
        """
        return self.X[idx], self.y[idx]

def split_data(folder_path, train_ratio=0.7, val_ratio=0.15, batch_size=100):
    """
    Split the dataset into training, validation, and test sets
    Args:
        folder_path (string): Path to the folder containing the dataset
        train_ratio (float): Ratio of the training set
        valid_ratio (float): Ratio of the validation set
        test_ratio (float): Ratio of the test set
    Returns:
        train_dataset, val_dataset, test_dataset
        each set is a tuple return by __getitem__ method of the CustomDataset class
    """
    base_dataset = CustomDataset(folder_path)
    train_size = int(train_ratio * len(base_dataset))
    val_size = int(val_ratio * len(base_dataset))
    test_size = len(base_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(base_dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)
    return train_loader, val_loader, test_loader


# create a dataset object
train_loader, val_loader, test_loader = split_data(folder_path='data')

# create a dataloader

# iterate over the dataloader
# for i, (X, y) in enumerate(train_loader):
#     print(f'Batch {i}:\nX: {X.shape}\ny: {y.shape}\n')
#     if i == 2:
#         break

# end time measurement
end = time.time()
print(f'Execution time: {end - start} seconds')