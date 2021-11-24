'''
Plot curve and required diagram
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math

import CNN_model
from CNN_model import CNN_Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
catagory_list = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def plot_curves(train_losses, train_accs, test_losses, test_accs):
    '''Plot learning curves and accuracy curves'''
    # Learning curves
    plt.plot(range(len(train_losses)), train_losses, label = "Train")
    plt.plot(range(len(test_losses)), test_losses, label = "Validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Learning Curves")
    plt.show()

    # Accuracy Curves
    plt.plot(range(len(train_accs)), train_accs, label = "Train")
    plt.plot(range(len(test_accs)), test_accs, label = "Validation")
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.show()

def plot_cnn_activation(image: np.ndarray, model: CNN_Network):
    image = np.expand_dims(image, axis = 0)
    image = np.expand_dims(image, axis = 0)
    image = torch.from_numpy(image)
    image = image.to(device)
    filt = model.filter
    
    # First conv
    image = model.conv1(image)
    image = model.ReLu1(image)
    activation = image.cpu().detach().numpy()

    n_row = math.floor(math.sqrt(filt))
    n_column = math.ceil(filt/n_row)
    fig, ax = plt.subplots(n_row, n_column, constrained_layout = True)
    for j in range(n_column):
        for i in range(n_row):
            # print(activation[0].shape)
            ax[i, j].imshow(activation[0][4*i + j], cmap = plt.get_cmap("gray"))
            ax[i, j].axis("off")
    plt.show()

def plot_resnet_activation(image, model):
    image = np.expand_dims(image, axis = 0)
    image = np.expand_dims(image, axis = 0)
    image = torch.from_numpy(image)
    image = image.to(device)
    image = model.conv1(image)
    relu = nn.ReLU(inplace = True)
    image = relu(image)
    activation = image.detach().cpu().numpy()

    fig,ax = plt.subplots(8, 8, constrained_layout = True)
    for j in range(8):
        for i in range(8):
            ax[i, j].imshow(activation[0][4*i + j], cmap = plt.get_cmap("gray"))
            ax[i, j].axis("off")
    plt.show()

def plot_prediction(X_arr: np.ndarray, Y_arr: np.ndarray, pred_arr: np.ndarray):
    fig, ax = plt.subplots(4, 4, constrained_layout = True)
    for i in range(4):
        for j in range(4):
            ax[i, j].imshow(X_arr[4*i + j], cmap = plt.get_cmap("gray"))
            if Y_arr[4*i + j] == pred_arr[4*i + j]:
                ax[i, j].text(14, 1, catagory_list[pred_arr[4*i + j]], horizontalalignment = 'center', verticalalignment = 'center', color = "lime", fontsize = 20)
            else:
                ax[i, j].text(14, 1, catagory_list[pred_arr[4*i + j]], horizontalalignment = 'center', verticalalignment = 'center', color = "red", fontsize = 20)
            ax[i, j].axis("off")
    
    plt.show()
