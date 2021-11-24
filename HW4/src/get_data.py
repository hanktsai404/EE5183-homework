'''
Financial Technology
Assignment 4
B07703014 蔡承翰
Due Jan 7th 2021

This module is intended to contain all data sources and do the preprocessing.
'''
import pandas as pd
import numpy as np
import talib
import yfinance as yf


def get_SP_500_all():
    df = pd.read_csv("../S_P.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index(df["Date"])
    df = df.drop(["Date"], axis = 1)
    df["K"], df["D"] = talib.STOCH(df["High"], df["Low"], df["Close"], fastk_period = 10)
    df["MA10"] = talib.SMA(df["Close"], timeperiod = 10)
    df["MA30"] = talib.SMA(df["Close"], timeperiod = 30)
    df = df.drop(["Adj Close"], axis = 1)
    # print(df.iloc[:50,])
    return df

def get_2020_SP_500():
    '''The return data is from 2019-11-01 to 2020-12-31'''
    SP_500 = yf.Ticker("^GSPC")
    df = SP_500.history(start = "2019-10-01", end = "2020-12-31").drop(["Dividends", "Stock Splits"], axis = 1)
    df["K"], df["D"] = talib.STOCH(df["High"], df["Low"], df["Close"], fastk_period = 10)
    df["MA10"] = talib.SMA(df["Close"], timeperiod = 10)
    df["MA30"] = talib.SMA(df["Close"], timeperiod = 30)
    return df

def preprocess(df):
    df = df.dropna()
    train_df = df.loc[:"2018-01-01"]
    test_df = df.loc["2018-01-01":]
    train_y_df = train_df[["Close"]]
    train_x_df = (train_df - train_df.min())/(train_df.max() - train_df.min())
    test_y_df = test_df[["Close"]]
    test_x_df = (test_df - test_df.min())/(test_df.max() - test_df.min())
    return train_x_df, train_y_df, test_x_df, test_y_df



if __name__ == "__main__":
    get_SP_500_all()