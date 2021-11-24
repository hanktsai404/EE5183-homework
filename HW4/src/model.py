'''
Financial Technology
Assignment 4
B07703014 蔡承翰
Due Jan 7th 2021

RNN and LSTM model
'''

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 9
TIME_STEP = 30

EPOCH = 20

class RNN_LSTM_GRU(nn.Module):
    def __init__(self, is_LSTM = False, is_GRU = False):
        super(RNN_LSTM_GRU, self).__init__()
        if is_LSTM:
            self.mod = nn.LSTM(input_size = INPUT_SIZE, hidden_size = 32, num_layers = 3, batch_first = True)
        elif is_GRU:
            self.mod = nn.GRU(input_size = INPUT_SIZE, hidden_size = 32, num_layers = 3, batch_first = True)
        else:
            self.mod = nn.RNN(input_size = INPUT_SIZE, hidden_size = 32, num_layers = 3, batch_first = True)
        self.out = nn.Linear(32, 1)
    
    def forward(self, x):
        r_out, h_n = self.mod(x, None)
        out = self.out(r_out[:,-1,:])
        return out

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X_arr, Y_arr):
        self.X = torch.from_numpy(X_arr)
        self.Y = torch.from_numpy(Y_arr)
    
    def __getitem__(self, index):
        return self.X[index:index + TIME_STEP], self.Y[index + TIME_STEP]
    
    def __len__(self):
        return len(self.Y) - TIME_STEP


def model_fit(trainX_arr, trainY_arr, testX_arr, testY_arr, is_LSTM = False, is_GRU = False):
    train_dataset = Dataset(trainX_arr, trainY_arr)
    test_dataset = Dataset(testX_arr, testY_arr)
    trainloader = DataLoader(train_dataset, batch_size = 128, shuffle = False, pin_memory = True)
    testloader = DataLoader(test_dataset, batch_size = 128, shuffle = False, pin_memory = True)

    net = RNN_LSTM_GRU(is_LSTM, is_GRU)
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    criterion = nn.MSELoss()
    net.to(device)

    train_losses = []
    test_losses = []

    for epoch in range(EPOCH):
        train_loss = 0
        for step, (x, y) in enumerate(trainloader):
            x = x.to(device)
            x = x.float()
            y = y.to(device)
            y = y.float()
            optimizer.zero_grad()
            # forward prop
            y_hat = net(x)
            loss = criterion(y_hat, y)
            # backward prop
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss)

        test_loss = 0
        for step, (x, y) in enumerate(testloader):
            x = x.to(device)
            x = x.float()
            y = y.to(device)
            y = y.float()
            y_hat = net(x)
            loss = criterion(y_hat, y)
            test_loss += loss.item()
        test_losses.append(test_loss)
        print("Epoch: " + str(epoch) + "\tTrain Loss: " + str(round(train_loss, 4)) + "\tValidation Loss: " + str(round(test_loss, 4)))
    
    return train_losses, test_losses, net

def prediction(X_arr, Y_arr, net, maximum, minimum):
    dataset = Dataset(X_arr, Y_arr)
    dataloader = DataLoader(dataset, batch_size = len(dataset), shuffle = False, pin_memory = True)
    preds = 0
    for step, (x, y) in enumerate(dataloader):
        x = x.to(device)
        x = x.float()
        preds = net(x)
    preds = preds.detach().cpu().numpy()
    preds = preds*(maximum - minimum) + minimum
    preds = np.transpose(preds).tolist()[0]
    preds = ([None]*30) + preds
    return preds



if __name__ == "__main__":
    pass