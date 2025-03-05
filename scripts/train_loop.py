import os
import torch
import torch.nn as nn
import torch.optim as optim
from scripts.custom_dataset import *

# set global parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = os.cpu_count()

# define the neural network
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(50, 128) # 1st fully connected layer (dense layer)
        self.fc2 = nn.Linear(128, 64) # 2nd fully connected layer (dense layer)
        self.fc3 = nn.Linear(64, 1) # 3rd fully connected layer (dense layer)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        output = torch.sigmoid(x)
        return output

# initialize the model
model = ToyModel().to(device)

# defining the loss function and optimiser
loss_fn = nn.BCELoss() # binary cross entropy loss
optimiser = optim.SGD(model.parameters(), lr=0.01)

def train_one_epoch(model, train_loader, val_loader, optimiser, loss_fn):
    running_train_loss = 0.0
    running_val_loss = 0.0
    last_loss = 0.0
    model.train()  # Set the model to training mode
    for i, (X_train, y_train) in enumerate(train_loader):
        # each iteration is a batch of data X, y
        X_train, y_train = X_train.to(device), y_train.to(device) # move data to device

        # set optimiser gradients to zero
        optimiser.zero_grad()

        # forward pass
        train_output = model(X_train)

        # calculate train loss
        train_loss = loss_fn(train_output, y_train.view(-1, 1))
        last_train_loss = running_train_loss / len(train_loader)

        # Validation step before backward and step
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for i, (X_val, y_val) in enumerate(val_loader):
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_output = model(X_val)
                val_loss += loss_fn(val_output, y_val.view(-1, 1)).item()
        
        model.train()  # Set the model back to training mode
        # backward pass
        train_loss.backward()

        # update weights
        optimiser.step()

        # Gather information and write       
        running_train_loss += train_loss.item()
        running_val_loss += val_loss
        last_val_loss = running_val_loss / len(val_loader)

    print(f'Training loss: {last_train_loss:.4f}, Validation loss: {last_val_loss:.4f}')

    return last_loss, val_loss

train_loader, val_loader, test_loader = split_data(folder_path='data')

train_one_epoch(model, train_loader, val_loader, optimiser, loss_fn)