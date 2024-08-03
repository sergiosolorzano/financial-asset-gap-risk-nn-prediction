from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf

import torch
print(torch.__version__)

def plot_weights_gradients(weights_dict, gradients_dict, epoch):
    for name, weight_list in weights_dict.items():
        plt.figure(figsize=(10, 6))
        plt.title(f"Epoch {epoch + 1} - Weights {name}")
        
        for i, w in enumerate(weight_list):
            plt.plot(w.flatten())
        
        plt.xlabel('Weight Index')
        plt.ylabel('Weight Value')
        plt.legend(loc="upper right")
        plt.show()

    for name, gradient_list in gradients_dict.items():
        plt.figure(figsize=(10, 6))
        plt.title(f"Epoch {epoch + 1} - Gradients {name}")
        
        for i, g in enumerate(gradient_list):
            plt.plot(g.flatten())
        
        plt.xlabel('Gradient Index')
        plt.ylabel('Gradient Value')
        plt.legend(loc="upper right")
        plt.show()

def scatter_diagram_onevar(stack_input):
    torch.set_printoptions(threshold=torch.inf)
    # output_string = f"stack_input {stack_input}"
    # with open(f'stack_input.txt', 'w') as file:
    #     file.write('\n\n' + output_string)
    reshaped_stack_input = stack_input.view(stack_input.size(0), stack_input.size(1), -1)
    
    means =[]
    for batch_idx in range(reshaped_stack_input.size(0)):
    #loop through images of each channel
        for chann in range(reshaped_stack_input.size(1)):
            var1_chann_image = reshaped_stack_input[batch_idx, chann, :]
            #print("var1channelimage",var1_chann_image)
            mean_value = torch.mean(var1_chann_image).item()
            means.append(mean_value)

    plt.figure(figsize=(10, 6))
    df = pd.DataFrame({
        'input_mean': means
    })
    #print("input mean",df)
    plt.scatter(df.index, df['input_mean'], c='red', marker='x', label='Input Mean Values')
    plt.xlabel('Input Mean Values')
    plt.ylabel('Values')
    plt.title('Scatter Diagram of Input Mean Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def scatter_diagram_twovar(test_stock_ticker,train_stock_ticker,var1, var2):
    #torch.set_printoptions(threshold=torch.inf)
    torch.set_printoptions(threshold=torch.inf)
    reshaped_test_stack_input = var1.view(var1.size(0), var1.size(1), -1)
    reshaped_train_stock_ticker = var2.view(var2.size(0), var2.size(1), -1)
    
    test_means =[]
    train_means =[]
    for batch_idx in range(reshaped_test_stack_input.size(0)):
    #loop through images of each channel
        for chann in range(reshaped_test_stack_input.size(1)):
            var1_chann_image = reshaped_test_stack_input[batch_idx, chann, :]
            var2_chann_image = reshaped_train_stock_ticker[batch_idx, chann, :]
            var1_mean_value = torch.mean(var1_chann_image).item()
            var2_mean_value = torch.mean(var2_chann_image).item()
            test_means.append(var1_mean_value)
            train_means.append(var2_mean_value)

    plt.figure(figsize=(10, 6))
    df = pd.DataFrame({
        'test_mean': test_means,
        'train_mean': train_means
    })
    #print("input mean",df)
    plt.scatter(df.index, df['test_mean'], c='red', marker='x', label=f'{test_stock_ticker} Mean Values')
    plt.scatter(df.index, df['train_mean'], c='blue', marker='x', label=f'{train_stock_ticker} Mean Values')
    plt.xlabel('Input Mean Values')
    plt.ylabel('Values')
    plt.title('Scatter Diagram of Test and Train Stocks Input Mean Values')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
    # var1_mean = var1.view(64, -1).mean(1, keepdim=True)
    # var2_mean = var2.view(64, -1).mean(1, keepdim=True)

    # df = pd.DataFrame({
    #     'var1_mean': var1_mean.cpu().numpy().flatten(),
    #     'var2_mean': var2_mean.cpu().numpy().flatten()
    # })

    # plt.figure(figsize=(10, 6))
    # plt.scatter(df.index, df['var1_mean'], c='red', marker='x', label=f'{test_stock_ticker} Mean Values')
    # plt.scatter(df.index, df['var2_mean'], c='blue', marker='x', label=f'{train_stock_ticker} Mean Values')


    # # Labeling the plot
    # plt.xlabel(f'{test_stock_ticker} and {train_stock_ticker} Mean Values')
    # plt.ylabel('Values')
    # plt.title(f'Scatter Diagram of {test_stock_ticker} and {train_stock_ticker} Mean Values')
    # plt.legend()
    # plt.grid(True)

    # # Show the plot
    # plt.show()

def compare_stocks(index_ticker, stock_ticker, stock_dataset, start_date, end_date):
    index_data = yf.download(index_ticker, start=start_date, end=end_date, interval='1d')

    stock_data = stock_dataset.dropna()
    index_data = index_data.dropna()

    stock_data = stock_data[stock_data.index <= end_date]
    index_data = index_data[index_data.index <= end_date]

    #rebase
    stock_rebased = stock_data / stock_data.iloc[0] * 100
    index_rebased = index_data / index_data.iloc[0] * 100

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    axs[0, 0].plot(stock_rebased.index, stock_rebased['Open'], label=f'{stock_ticker} Open Price', color='g')
    axs[0, 0].plot(stock_rebased.index, stock_rebased['Close'], label=f'{stock_ticker} Close Price', color='m')
    axs[0, 0].plot(index_rebased.index, index_rebased['Close'], label=f'{index_ticker} Close Price', color='m', linestyle='--')
    axs[0, 0].set_title('Open and Close Prices')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(stock_rebased.index, stock_rebased['Close'], label=f'{stock_ticker} Close Price', color='m')
    axs[0, 1].plot(stock_rebased.index, stock_rebased['High'], label=f'{stock_ticker} High Price', color='b')
    axs[0, 1].plot(index_rebased.index, index_rebased['Close'], label=f'{index_ticker} Close Price', color='m', linestyle='--')
    axs[0, 1].set_title('Close and High Prices')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(stock_rebased.index, stock_rebased['Close'], label=f'{stock_ticker} Close Price', color='m')
    axs[1, 0].plot(stock_rebased.index, stock_rebased['Low'], label=f'{stock_ticker} Low Price', color='r')
    axs[1, 0].plot(index_rebased.index, index_rebased['Close'], label=f'{index_ticker} Close Price', color='m', linestyle='--')
    axs[1, 0].set_title('Close and Low Prices')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(stock_rebased.index, stock_rebased['High'], label=f'{stock_ticker} High Price', color='b')
    axs[1, 1].plot(stock_rebased.index, stock_rebased['Low'], label=f'{stock_ticker} Low Price', color='r')
    axs[1, 1].plot(index_rebased.index, index_rebased['High'], label=f'{index_ticker} High Price', color='b', linestyle='--')
    axs[1, 1].set_title('High and Low Prices')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_image_correlations(series_correlations, mean_correlation):
    # Plot the correlations
    plt.figure(figsize=(12, 8))
    sns.histplot(series_correlations, kde=True, bins=30)
    plt.axvline(mean_correlation, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_correlation:.2f}')
    plt.title('Distribution of Image Series Correlations')
    plt.xlabel('Image Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.show()
    