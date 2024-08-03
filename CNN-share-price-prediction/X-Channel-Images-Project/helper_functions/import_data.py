from __future__ import print_function

import os
import sys
import glob
import time
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf


def import_dataset(ticker, start_date, end_date):

    dataset = yf.download(ticker, start=start_date, end=end_date, interval='1d')

    print("num rows",dataset.shape[0])
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', None)
    dataset = dataset.dropna().dropna()
    dataset.dropna(how='any', inplace=True)
    print("Num rows for df Close col",len(dataset['Close']))
    print(dataset.columns)
    dataset = dataset.reset_index()
    #reorder to split the data to train and test
    desired_order = ['Date','Open', 'Close', 'High', 'Low']
    if 'Date' in dataset.columns:
        dataset = dataset[desired_order]
    else:
        print("Column 'Date' is missing.")
    dataset = dataset.set_index('Date')
    print(dataset.head())
    print("day count",dataset.index.max()-dataset.index.min())

    #print(dataset.iloc[521],dataset.iloc[522],dataset.iloc[523])
    # pd.reset_option('display.max_rows')
    # pd.reset_option('display.max_columns')
    # pd.reset_option('display.width')
    # pd.reset_option('display.max_colwidth')

    return dataset

