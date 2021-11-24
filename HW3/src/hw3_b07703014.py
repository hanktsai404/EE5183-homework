'''
Financial Technology
Assignment 3
B07703014 蔡承翰
Due Dec 12th 2020
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import getdata
import CNN_model
import plot

totalX_arr, totalY_arr = getdata.get_data()
trainX_arr, trainY_arr, testX_arr, testY_arr = getdata.split_data(totalX_arr, totalY_arr)

plt.imshow(testX_arr[0], cmap = plt.get_cmap("gray"))
plt.axis("off")
plt.show()

print("Do you want to do grid search on stride and filter? (y/n)")
instruct = input()
if instruct == "y":
    stride, size = CNN_model.train_cnn(trainX_arr, trainY_arr, testX_arr, testY_arr)
else:
    stride = 2
    size = 5

filt = 16
train_losses, train_accs, test_losses, test_accs, model = CNN_model.model_fit(stride, filt, size, trainX_arr, trainY_arr, testX_arr, testY_arr)
preds = CNN_model.prediction(trainX_arr[0:2], trainY_arr[0:2], model)
print("First few true Y on the train set")
print(trainY_arr[0:2])
print("First few prediction on the train set")
print(preds)

while True:
    print("Please enter instruction")
    print("0: CNN Learning curve and accuracy curve\n1: CNN Activation plot\n2: CNN Prediction plot\n3: Train Resnet\n4: Resnet Leaning curve and accuracy curve")
    print("5: Resnet Activation plot\n6: Resnet Prediction plot\n9: Exit")
    instruct = input()
    print()
    if instruct == "0":
        plot.plot_curves(train_losses, train_accs, test_losses, test_accs)
    elif instruct == "1":
        plot.plot_cnn_activation(trainX_arr[0], model)
    elif instruct == "2":
        preds = CNN_model.prediction(testX_arr[0:16], testY_arr[0:16], model)
        plot.plot_prediction(testX_arr[0:16], testY_arr[0:16], preds)
    elif instruct == "3":
        res_train_losses, res_train_accs, res_test_losses, res_test_accs, res_model = CNN_model.resnet_fit(trainX_arr, trainY_arr, testX_arr, testY_arr)
        preds = CNN_model.prediction(trainX_arr[0:2], trainY_arr[0:2], res_model)
        print("First few true Y on the train set")
        print(trainY_arr[0:2])
        print("First few prediction on the train set")
        print(preds)
    elif instruct == "4":
        plot.plot_curves(res_train_losses, res_train_accs, res_test_losses, res_test_accs)
    elif instruct == "5":
        plot.plot_resnet_activation(trainX_arr[0], res_model)
    elif instruct == "6":
        preds = CNN_model.prediction(testX_arr[0:16], testY_arr[0:16], res_model)
        plot.plot_prediction(testX_arr[0:16], testY_arr[0:16], preds)
    else:
        break

