import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scripts.generate_dataset import *
from scripts.custom_dataset import *
from scripts.train_loop import *

# set global parameters
rng = np.random.default_rng(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = os.cpu_count()
print(f'Using device: {device}')
print(f'Number of workers: {num_workers}')


num_samples = 1000
num_features = 50
folder_path = '/mnt/d/torch-playground/data'
extension = 'csv'
num_files = 5

train_ratio = 0.7
val_ratio = 0.15

batch_size = 100

    
def main():
    # generate multiple datasets
    print(f'Start generating datasets...')
    generate_dataset()
    print(f'Finished generating datasets...')
    
    # split the dataset into training, validation, and test sets
    print(f'Start splitting datasets...')
    train_loader, val_loader, test_loader = split_data(folder_path, train_ratio, val_ratio, batch_size)
    # print(f'Finished splitting datasets...')
    # for i, (X, y) in enumerate(train_loader):
    #     print(f'Batch {i}:\nX: {X.shape}\ny: {y.shape}\n')
    #     if i == 2:
    #         break

    model = ToyModel().to(device)
    loss_fn = nn.BCELoss()
    optimiser = optim.SGD(model.parameters(), lr=0.01)

    train_one_epoch(model, train_loader, val_loader, optimiser, loss_fn)

if __name__ == '__main__':
    main()



