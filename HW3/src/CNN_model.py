'''
CNN model for training and grid search
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN_Network(nn.Module):
    def __init__(self, stride: int, filt: int, size: int):
        super(CNN_Network, self).__init__()
        self.stride = stride
        self.filter = filt
        
        # Model: 3 layer CNN
        # Conv-1
        self.conv1 = nn.Conv2d(1, self.filter, kernel_size = size, stride = self.stride, padding = (1, 1))
        self.ReLu1 = nn.ReLU(inplace = True)

        # Conv-2
        self.conv2 = nn.Conv2d(self.filter, self.filter, kernel_size = size, stride = self.stride, padding = (1,1))
        self.ReLu2 = nn.ReLU(inplace = True)

        # Conv-3
        self.conv3 = nn.Conv2d(self.filter, self.filter, kernel_size = size, stride = self.stride, padding = (1, 1))
        self.ReLu3 = nn.ReLU(inplace = True)

        # Pooling
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # Linear
        self.fc = nn.Linear(self.filter, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.ReLu1(x)
        x = self.conv2(x)
        x = self.ReLu2(x)
        x = self.conv3(x)
        x = self.ReLu3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print(x.size())
        return x

def model_fit(stride: int, filt: int, size: int, trainX_arr, trainY_arr, testX_arr, testY_arr):
    train_dataset = TensorDataset(torch.from_numpy(trainX_arr), torch.from_numpy(trainY_arr).type(torch.LongTensor))
    test_dataset = TensorDataset(torch.from_numpy(testX_arr), torch.from_numpy(testY_arr).type(torch.LongTensor))
    trainloader = DataLoader(train_dataset, batch_size = 1024, shuffle = False, pin_memory = True)
    testloader = DataLoader(test_dataset, batch_size = 1024, shuffle = False, pin_memory = True)

    net = CNN_Network(stride, filt, size)
    net.to(device)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    input_shape = (-1, 1, 28, 28)
    for epoch in range(25):  # 25 epochs
        train_loss = 0
        correct_train = 0
        total_train = 0
        # train data
        for b_num, (x, y) in enumerate(trainloader):
            x = x.to(device)
            x = x.float()
            x = x.view(input_shape)
            y = y.to(device)
            optimizer.zero_grad()
            # forward prop
            y_hat = net(x)
            loss = criterion(y_hat, y)
            # backward prop
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # Compute accuaracy
            prediction = torch.max(y_hat.data, 1)[1]
            total_train += len(y)
            correct_train += (prediction == y).float().sum()
        
        train_losses.append(train_loss)
        train_acc = (correct_train / total_train).item()
        train_accs.append(train_acc)

        # validation
        test_loss = 0
        correct_test = 0
        total_test = 0
        for b_num, (x, y) in enumerate(testloader):
            x = x.to(device)
            x = x.float()
            x = x.view(input_shape)
            y = y.to(device)
            y_hat = net(x)
            loss = criterion(y_hat, y)
            test_loss += loss.item()
            prediction = torch.max(y_hat.data, 1)[1]
            total_test += len(y)
            correct_test += (prediction == y).float().sum()
        
        test_losses.append(test_loss)
        test_acc = (correct_test / total_test).item()
        test_accs.append(test_acc)
        print("Epoch: " + str(epoch) + "\tTrain loss: " + str(round(train_loss, 3)) + "\tTrain accuracy: " + str(round(train_acc, 3)) + "\tTest loss: " + str(round(test_loss, 3)) + "\tTest accuracy: " + str(round(test_acc, 3)))
    
    return train_losses, train_accs, test_losses, test_accs, net

def train_cnn(trainX_arr, trainY_arr, testX_arr, testY_arr):
    '''Train CNN model, returning the prediction on trainset and testset, losses and accuracies of each epoch'''

    strides = [1, 2]
    filt = 16
    sizes = [1, 3, 5]

    best_test_acc = 100.0
    best_stride = 0
    best_size = 0
    for stride in strides:
        for size in sizes:
            train_losses, train_accs, test_losses, test_accs, net = model_fit(stride, filt, size, trainX_arr, trainY_arr, testX_arr, testY_arr)
            if test_accs[-1] < best_test_acc:
                best_test_acc = test_accs[-1]
                best_stride = stride
                best_size = size
    
    print("Training result\n" + "Best stride: " + str(best_stride) + "\tBest filter " + str(size))
    return best_stride, best_size

def prediction(testX_arr, testY_arr, net):
    size = testX_arr.shape[0]

    dataset = TensorDataset(torch.from_numpy(testX_arr), torch.from_numpy(testY_arr).type(torch.LongTensor))
    loader = DataLoader(dataset, batch_size = size, shuffle = False, pin_memory = True)

    prediction = 0
    input_shape = (-1, 1, 28, 28)
    for b_num, (x, y) in enumerate(loader):
        x = x.to(device)
        x = x.float()
        x = x.view(input_shape)
        y = y.to(device)
        y_hat = net(x)
        prediction = torch.max(y_hat.data, 1)[1]
    
    # print(prediction)
    return prediction.detach().cpu().numpy()

def build_resnet():
    '''get resnet model'''
    resnet_model = torchvision.models.resnet18(num_classes = 10)
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
    resnet_model.loss_criterion = nn.CrossEntropyLoss()
    return resnet_model

def resnet_fit(trainX_arr: np.ndarray, trainY_arr: np.ndarray, testX_arr: np.ndarray, testY_arr: np.ndarray):
    '''train a resnet model'''
    train_dataset = TensorDataset(torch.from_numpy(trainX_arr), torch.from_numpy(trainY_arr).type(torch.LongTensor))
    test_dataset = TensorDataset(torch.from_numpy(testX_arr), torch.from_numpy(testY_arr).type(torch.LongTensor))
    trainloader = DataLoader(train_dataset, batch_size = 1024, shuffle = False, pin_memory = True)
    testloader = DataLoader(test_dataset, batch_size = 1024, shuffle = False, pin_memory = True)

    net = build_resnet()
    criterion = net.loss_criterion
    net.to(device)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr = 0.01)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    input_shape = (-1, 1, 28, 28)
    for epoch in range(25):  # 25 epochs
        train_loss = 0
        correct_train = 0
        total_train = 0
        # train data
        for b_num, (x, y) in enumerate(trainloader):
            x = x.to(device)
            x = x.float()
            x = x.view(input_shape)
            y = y.to(device, non_blocking = True)
            optimizer.zero_grad()
            # forward prop
            y_hat = net(x)
            loss = criterion(y_hat, y)
            # backward prop
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # Compute accuaracy
            prediction = torch.max(y_hat.data, 1)[1]
            total_train += len(y)
            correct_train += (prediction == y).float().sum()
            # print(correct_train)
            print("total train: " + str(total_train))
        
        train_losses.append(train_loss)
        train_acc = (correct_train / total_train).item()
        train_accs.append(train_acc)
        print(train_acc)

        # validation
        test_loss = 0
        correct_test = 0
        total_test = 0
        for b_num, (x, y) in enumerate(testloader):
            x = x.to(device)
            x = x.float()
            x = x.view(input_shape)
            y = y.to(device)
            y_hat = net(x)
            loss = criterion(y_hat, y)
            test_loss += loss.item()
            prediction = torch.max(y_hat.data, 1)[1]
            total_test += len(y)
            correct_test += (prediction == y).float().sum()
        
        test_losses.append(test_loss)
        test_acc = (correct_test / total_test).item()
        test_accs.append(test_acc)
        print(test_acc)
        print("Epoch: " + str(epoch) + "\tTrain loss: " + str(round(train_loss, 3)) + "\tTrain accuracy: " + str(round(train_acc, 3)) + "\tTest loss: " + str(round(test_loss, 3)) + "\tTest accuracy: " + str(round(test_acc, 3)))
    
    return train_losses, train_accs, test_losses, test_accs, net
