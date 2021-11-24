'''
Financial Technology
Assignment 4
B07703014 蔡承翰
Due Jan 7th 2021

This module is intended to contain all functions generate plots
'''

import matplotlib.pyplot as plt
import mplfinance as mpf
from talib import abstract
import pandas as pd
import model

def plot_candlestick(df):
    style = mpf.make_mpf_style(base_mpf_style = "charles", mavcolors = ["skyblue", "orange"])
    ma10 = mpf.make_addplot(df["MA10"], panel = 0)
    ma30 = mpf.make_addplot(df["MA30"], panel = 0)
    sto_k = mpf.make_addplot(df["K"], panel = 2, ylabel = "Percentage")
    sto_d = mpf.make_addplot(df["D"], panel = 2)
    fig, axes = mpf.plot(df, type = "candle", style = style,
                         volume = True, ylabel_lower = "Shares\nTraded",
                         addplot = [ma10, ma30, sto_k, sto_d], returnfig = True, show_nontrading = True)
    axes[0].legend(["10d MA", "30d MA"])
    axes[4].legend(["Stochastic K", "Stochastic D"])
    plt.show()

def plot_loss_curve(train_losses, test_losses, name):
    plt.plot(range(len(train_losses)), train_losses, label = "Train")
    plt.plot(range(len(test_losses)), test_losses, label = "Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(name + " Loss")
    plt.legend()
    plt.show()

def plot_prediction(preds_df, name, net, is_2020 = False):
    predsX_df = (preds_df - preds_df.min())/(preds_df.max() - preds_df.min())
    predsY_df = preds_df[["Close"]]
    predsY_max = predsY_df.max()[0]
    predsY_min = predsY_df.min()[0]
    predsY_df = (predsY_df - predsY_df.min())/(predsY_df.max() - predsY_df.min())
    preds = model.prediction(predsX_df.to_numpy(), predsY_df.to_numpy(), net, predsY_max, predsY_min)
    preds_df = preds_df[["Close"]]
    preds_df["Predict"] = preds
    if is_2020:
        preds_df = preds_df["2020-01-01":]
    else:
        preds_df = preds_df.loc["2018-01-01":]
    preds_df.plot(style = ["-", "--"], title = name)
    plt.show()
    