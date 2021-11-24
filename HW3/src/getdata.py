# import the necessary packages
from keras.datasets import fashion_mnist
import numpy as np

def get_data():
    '''Get totalX. totalY in ndarray and save them in csv, first part of codes are given'''
    print("[INFO] loading Fashion MNIST...")
    ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

    trainX = trainX.astype("float32") / 255  # Normalization
    testX = testX.astype("float32") / 255
    totalX = np.concatenate((trainX, testX), axis = 0)
    totalY = np.concatenate((trainY, testY), axis = 0)
    return totalX, totalY

def split_data(totalX, totalY):
    '''Split data into trainX, trainY, testX, testY in ndarray'''
    n_sample = totalY.shape[0]
    n_train_sample = round(n_sample * 0.8)  # 80% train / 20% test
    trainX = np.array([totalX[i,...] for i in range(n_train_sample)])
    trainY = np.array([totalY[i,...] for i in range(n_train_sample)])
    testX = np.array([totalX[i,...] for i in range(n_train_sample, n_sample)])
    testY = np.array([totalY[i,...] for i in range(n_train_sample, n_sample)])
    return trainX, trainY, testX, testY


if __name__ == "__main__":
    print("Module testing...")
    totalX, totalY = get_data()
    trainX, trainY, testX, testY = split_data(totalX, totalY)
    if trainX.shape == (56000, 28, 28) and trainY.shape == (56000,) and testX.shape == (14000, 28, 28) and testY.shape == (14000,):
        print("No error here")
    else:
        print("Error!")