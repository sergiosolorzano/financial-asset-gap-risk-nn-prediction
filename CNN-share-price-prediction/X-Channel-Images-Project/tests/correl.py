import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use(Parameters.matplotlib_use)
import matplotlib.pyplot as plt
import yfinance as yf

from parameters import Parameters

matplotlib.use(Parameters.matplotlib_use)

class StockParams:
    def __init__(self):
        self.train_stocks = []
        self.eval_stocks = []
        self.train_stock_tickers = ""
        self.eval_stock_tickers = ""
        # self.start_date = ''
        # self.end_date = ''

    def add_train_stock(self, ticker, start_date, end_date):
        stock_info = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        }
        self.train_stocks.append(stock_info)

    def add_eval_stock(self, ticker, start_date, end_date):
        stock_info = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        }
        self.eval_stocks.append(stock_info)

    def get_train_stocks(self):
        return self.train_stocks
    
    def get_eval_stocks(self):
        return self.eval_stocks
    
    def set_param_strings(self):
        train_stocks = self.get_train_stocks()
        eval_stocks = self.get_eval_stocks()
        for s in train_stocks:
            self.train_stock_tickers = "_".join([s['ticker'] for s in train_stocks])
        for s in eval_stocks:
            self.eval_stock_tickers = "_".join([s['ticker'] for s in eval_stocks])
        # self.start_date = train_stocks[0]['start_date']
        # self.end_date = train_stocks[0]['end_date']

def plot_train_eval_cross_correl_price_series(stocks, start_date, end_date):
    data_close = {}

    for s in stocks.get_train_stocks() + stocks.get_eval_stocks():
        dataset_df = yf.download(s['ticker'], start=start_date, end=end_date, interval='1d')
        dataset_df = dataset_df.dropna()
        #reset column to save to csv and mlflow schema
        dataset_df = dataset_df.reset_index()

        #reorder to split the data to train and test
        desired_order = ['Date','Open', 'Close', 'High', 'Low']
        if 'Date' in dataset_df.columns:
            dataset_df = dataset_df[desired_order]
        else:
            print("Column 'Date' is missing.")

        data_close[s['ticker']] = dataset_df['Close']

        # calc correl training datasets
        df_close = pd.DataFrame(data_close)
        if s['ticker']=='SICP':
            print("sicp",df_close)
        
    cross_corr_matrix = df_close.corr(method='spearman')
    print("Train & Eval set cross_corr_matrix",cross_corr_matrix)
        
stock_params = StockParams()

# Add stocks and set parameters
start_date='2021-12-05'
end_date='2023-01-25'
stock_params.add_train_stock('SIVBQ', start_date, end_date)
stock_params.add_train_stock('SICP', start_date, end_date)
stock_params.add_train_stock('CMA', start_date, end_date)
stock_params.add_eval_stock('WAL', start_date, end_date)
stock_params.add_eval_stock('ZION', start_date, end_date)
stock_params.add_eval_stock('PNC', start_date, end_date)
stock_params.add_eval_stock('ALLY', start_date, end_date)
stock_params.add_train_stock('FITB', start_date, end_date)
stock_params.add_eval_stock('RF', start_date, end_date)
# stock_params.add_eval_stock('CROX', start_date, end_date)
# stock_params.add_eval_stock('DIS', start_date, end_date)
# stock_params.add_eval_stock('UAL', start_date, end_date)
stock_params.add_eval_stock('JPM', start_date, end_date)

stock_params.set_param_strings()

plot_train_eval_cross_correl_price_series(stock_params, start_date, end_date)