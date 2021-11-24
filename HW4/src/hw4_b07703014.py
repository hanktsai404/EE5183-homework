'''
Financial Technology
Assignment 4
B07703014 蔡承翰
Due Jan 7th 2021

Main program
'''

import plot
import get_data
import model
import pandas as pd
import numpy as np


SP_500_all_df = get_data.get_SP_500_all()
trainX_df, trainY_df, testX_df, testY_df = get_data.preprocess(SP_500_all_df)
trainY_max = trainY_df.max()[0]
trainY_min = trainY_df.min()[0]
testY_max = testY_df.max()[0]
testY_min = testY_df.min()[0]
trainY_df = (trainY_df - trainY_df.min())/(trainY_df.max() - trainY_df.min())
testY_df = (testY_df - testY_df.min())/(testY_df.max() - testY_df.min())

# Data from 2020
SP_500_2020_df = get_data.get_2020_SP_500()



while True:
    print("Plese enter instruction:")
    print("0: plot klines accross 2019\n1: Train RNN model\n2: Train LSTM model\n3: Train GRU model\n9: Exit")
    instruct = input()
    if instruct == "0":
        SP_500_2019_df = SP_500_all_df.loc["2019-01-01":"2020-01-01"]
        plot.plot_candlestick(SP_500_2019_df)
    elif instruct == "1":
        train_losses, test_losses, net = model.model_fit(
            trainX_df.to_numpy(), trainY_df.to_numpy(), testX_df.to_numpy(), testY_df.to_numpy())
        plot.plot_loss_curve(train_losses, test_losses, "RNN")

        preds_df = SP_500_all_df.loc["2017-11-01":]
        plot.plot_prediction(preds_df, "RNN prediction 2018~2019", net)
        plot.plot_prediction(SP_500_2020_df, "RNN prediction 2020", net, is_2020 = True)
    elif instruct == "2":
        train_losses, test_losses, net = model.model_fit(
            trainX_df.to_numpy(), trainY_df.to_numpy(), testX_df.to_numpy(), testY_df.to_numpy(), is_LSTM = True)
        plot.plot_loss_curve(train_losses, test_losses, "LSTM")

        preds_df = SP_500_all_df.loc["2017-11-01":]
        plot.plot_prediction(preds_df, "LSTM prediction 2018~2019", net)
        plot.plot_prediction(SP_500_2020_df, "LSTM prediction 2020", net, is_2020 = True)
    elif instruct == "3":
        train_losses, test_losses, net = model.model_fit(
            trainX_df.to_numpy(), trainY_df.to_numpy(), testX_df.to_numpy(), testY_df.to_numpy(), is_GRU = True)
        plot.plot_loss_curve(train_losses, test_losses, "GRU")

        preds_df = SP_500_all_df.loc["2017-11-01":]
        plot.plot_prediction(preds_df, "GRU prediction 2018~2019", net)
        plot.plot_prediction(SP_500_2020_df, "GRU prediction 2020", net, is_2020 = True)
    else:
        break


