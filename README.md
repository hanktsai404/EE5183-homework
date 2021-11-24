# EE5183-homework
This repository is a collection of homeworks of the course EE5183 in NTU in the semester of fall, 2020. The course introduced a variety of deep learning (DL) models. There are four homeworks in this course. The topic and the summarized result can all be found in the pdf files in each folder. Brief introductions to each assignments are as follows.

## Homework 1: Linear Regression
The assignment asks students to perform linear regression on two datasets. The models include linear regression with and without bias term and Bayesian regression. Later, in this assignment I discussed the difference of performances in each model. The assignment was coded only in python default package, numpy/pandas and matplotlib package.

## Homework 2: Credit Card Fraud Detection
The assignment aimed to utilize DNN model to detect credit card fraud. Later on, several criterion was given to judge the performance of the model. The given data had gone through the PCA process.

# Homework 3: Image Classification
The assignment utilized CNN model to classify images of clothing products. After training the CNN model, I used resnet to train the model again, trying to improve the performance. The assignment concluded that using resnet is better than using CNN in this specific case.

# Homework 4: Stock Price Prediction
The close price of S&P 500 using data from 1994 to 2017. Firstly, the moving average chart and KD chart are plotted. Secondly, after training the RNN model, LSTM and GRU cells were later included in the model. Lastly, I tested the model with 2020 data. The concluding remark in this assignment is that though we can achieve low validation loss, we cannot rule out the randomness of the stock prices. The prediction value depends heavily on the last observation. That is, the pattern of stock prices likely follows unit root process (or even martingale process). The prediction cannot benefit investor because the price of S&P 500 is still a "fair game".
 
The models in homework 2,3,4 were built by PyTorch and Torchvision packages. For more detail introduction to the works please refer to the pdf files in each folders
