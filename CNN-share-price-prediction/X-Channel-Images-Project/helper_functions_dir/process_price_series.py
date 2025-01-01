from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
from fastdtw import fastdtw

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random as rand

import yfinance as yf

from parameters import Parameters
import pipeline_data

import mlflow

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import helper_functions as helper_functions

matplotlib.use(Parameters.matplotlib_use)

def log_rebase_dataset(stocks):
    data_close = {}
    merged_df = None

    for s in stocks.get_train_stocks() + stocks.get_eval_stocks():
        print("Log Rebasing Stock",s['ticker'])
        dataset_df = yf.download(s['ticker'], start=s['start_date'], end=s['end_date'], interval='1d')
        dataset_df = dataset_df.dropna()
        dataset_df = dataset_df.reset_index()

        desired_order = ['Date','Open', 'Close', 'High', 'Low']
        if 'Date' in dataset_df.columns:
            dataset_df = dataset_df[desired_order]
        else:
            print("Column 'Date' is missing.")

        #data_close[s['ticker']] = dataset_df['Close']

        date_col = dataset_df['Date'].copy()
        dataset_df.drop(columns=['Date'], inplace=True)
        rebased_df = pipeline_data.remap_to_log_returns(dataset_df, None)
        rebased_df['Date'] = date_col.values[:-1]
        data_close[s['ticker']] = rebased_df['Close']

        if merged_df is None:
            merged_df = pd.DataFrame({'Date': rebased_df['Date'], s['ticker']: rebased_df['Close']})
        else:
            merged_df = merged_df.merge(
                pd.DataFrame({'Date': rebased_df['Date'], s['ticker']: rebased_df['Close']}), 
                on='Date', 
                how='outer' 
            )

    #print("MERGED",merged_df)
    return data_close, merged_df