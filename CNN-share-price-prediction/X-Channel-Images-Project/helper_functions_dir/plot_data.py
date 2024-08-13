from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import random as rand

import threading

import yfinance as yf

from parameters import Parameters

import mlflow

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import helper_functions as helper_functions

import torch
print(torch.__version__)

def plot_weights_gradients(weights_dict, gradients_dict, epoch):
    for name, weight_list in weights_dict.items():
        fig = plt.figure(figsize=(10, 6))
        plt.title(f"Epoch {epoch + 1} - Weights {name}")
        
        for i, w in enumerate(weight_list):
            plt.plot(w.flatten(), label=f'Weight {i}')
        
        plt.xlabel('Weight Index')
        plt.ylabel('Weight Value')
        plt.legend(loc="upper right")

        image_path = helper_functions.get_next_image_number()
        #write image to md
        if Parameters.save_runs_to_md:
            plt.savefig(image_path, dpi=300)
            helper_functions.write_to_md("plot_weights<p>",image_path)

        mlflow.log_figure(fig, image_path)

        #plt.show()
        plt.close(fig)

    for name, gradient_list in gradients_dict.items():
        fig = plt.figure(figsize=(10, 6))
        plt.title(f"Epoch {epoch + 1} - Gradients {name}")
        
        for i, g in enumerate(gradient_list):
            plt.plot(g.flatten(), label=f'Gradient {i}')
        
        plt.xlabel('Gradient Index')
        plt.ylabel('Gradient Value')
        plt.legend(loc="upper right")

        image_path = helper_functions.get_next_image_number()
        #write image to md
        if Parameters.save_runs_to_md:
            plt.savefig(image_path, dpi=300)
            helper_functions.write_to_md("plot_gradients<p>",image_path)

        mlflow.log_figure(fig, image_path)

        #plt.show()
        plt.close(fig)


def scatter_diagram_onevar_plot_mean(stack_input, stock_ticker):
    torch.set_printoptions(threshold=torch.inf)
    reshaped_stack_input = stack_input.view(stack_input.size(0), stack_input.size(1), -1)
    
    means =[]
    for batch_idx in range(reshaped_stack_input.size(0)):
    #loop through images of each channel
        for chann in range(reshaped_stack_input.size(1)):
            var1_chann_image = reshaped_stack_input[batch_idx, chann, :]
            #print("var1channelimage",var1_chann_image)
            mean_value = torch.mean(var1_chann_image).item()
            means.append(mean_value)

    fig = plt.figure(figsize=(10, 6))
    df = pd.DataFrame({
        f'input_mean {stock_ticker}': means
    })
    #print("input mean",df)
    plt.scatter(df.index, df[f'input_mean {stock_ticker}'], c='red', marker='x', label=f'{stock_ticker} Image Input Mean Values')
    plt.xlabel(f'{stock_ticker} Image Input Mean Values')
    plt.ylabel('Values')
    plt.title(f'Scatter Diagram of {stock_ticker} Image Input Mean Values')
    plt.legend(loc="upper right")
    plt.grid(True)

    image_path = helper_functions.get_next_image_number()
    #write image to md
    if Parameters.save_runs_to_md:
        plt.savefig(image_path, dpi=300)
        helper_functions.write_to_md("scatter_diagram_onevar_plot_mean<p>",image_path)

    mlflow.log_figure(fig, image_path)

    #plt.show()
    plt.close(fig)


def scatter_diagram_twovar_plot_mean(test_stock_ticker,train_stock_ticker,var1, var2):
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

    fig = plt.figure(figsize=(10, 6))
    df = pd.DataFrame({
        f'{test_stock_ticker} test_mean': test_means,
        f'{train_stock_ticker} train_mean': train_means
    })
    #print("input mean",df)
    plt.scatter(df.index, df[f'{test_stock_ticker} test_mean'], c='red', marker='x', label=f'{test_stock_ticker} Mean Values')
    plt.scatter(df.index, df[f'{train_stock_ticker} train_mean'], c='blue', marker='x', label=f'{train_stock_ticker} Mean Values')
    plt.xlabel(f'{train_stock_ticker} and {test_stock_ticker} Input Mean Values')
    plt.ylabel('Values')
    plt.title(f'Scatter Diagram of {test_stock_ticker} and {train_stock_ticker} Image Input Mean Values')
    plt.legend(loc="upper right")
    plt.grid(True)

    image_path = helper_functions.get_next_image_number()
    #write image to md
    if Parameters.save_runs_to_md:
        plt.savefig(image_path, dpi=300)
        helper_functions.write_to_md("scatter_diagram_twovar_plot_mean<p>",image_path)

    mlflow.log_figure(fig, image_path)
    
    #plt.show()
    plt.close(fig)

    
def plot_price_comparison_stocks(index_ticker,train_stock_ticker,stock_dataset_df, start_date, end_date):
    fig, image_path = compare_stocks(index_ticker,train_stock_ticker,stock_dataset_df, start_date, end_date)
    
    return fig, image_path
    
def compare_stocks(index_ticker, stock_ticker, stock_dataset, start_date, end_date):

    index_data = yf.download(index_ticker, start=start_date, end=end_date, interval='1d')

    stock_data = stock_dataset.dropna()
    index_data = index_data.dropna()

    stock_data = stock_data[stock_data.index <= end_date]
    index_data = index_data[index_data.index <= end_date]

    #rebase
    stock_rebased = stock_data / stock_data.iloc[0] * 100
    index_rebased = index_data / index_data.iloc[0] * 100

    fig, axs = plt.subplots(2, 2, figsize=(Parameters.plt_image_size[0], Parameters.plt_image_size[1]))

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

    image_path = helper_functions.get_next_image_number()
    #write image to md
    if Parameters.save_runs_to_md:
        plt.savefig(image_path, dpi=300)
        plt.close(fig)
        helper_functions.write_to_md("plot_price_comparison_stocks<p>",image_path)

    mlflow.log_figure(fig, image_path)

    #plt.show()

    return fig, image_path

def plot_image_correlations(series_correlations, mean_correlation):
    # Plot the correlations
    fig = plt.figure(figsize=(12, 8))
    sns.histplot(series_correlations, kde=True, bins=30)
    plt.axvline(mean_correlation, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_correlation:.2f}')
    plt.title('Distribution of Trained vs Test Stocks Input Image Series Correlations')
    plt.xlabel('Image Correlation Coefficient')
    plt.ylabel('Frequency')

    image_path = helper_functions.get_next_image_number()
    #write image to md
    if Parameters.save_runs_to_md:
        plt.savefig(image_path, dpi=300)
        helper_functions.write_to_md("plot_image_correlations<p>",image_path)

    mlflow.log_figure(fig, image_path)

    #plt.show()
    plt.close(fig)

    
def quick_view_images(images_array, cols_used_count, cols_used):
    
    # Plot the first image of each column
    fig, axes = plt.subplots(nrows=1, ncols=cols_used_count, figsize=(20, 6))
    for ax in axes:
        ax.set_aspect('equal')

    print("shape images array",images_array.shape,"shape image",images_array[0][0][0][0].shape)
    for i in range(cols_used_count):
        axes[i].imshow(images_array[0][0][i][0], cmap='hot')
        axes[i].set_title(f"Column {cols_used[i]} ")

    #average first image of all features
    average_images = []
    for i in range(cols_used_count):
        average_images.append(images_array[0][0][i][0])

    average_image = np.mean(average_images, axis=0)

    # Hide axes
    for ax in axes:
        ax.axis('off')

    # Plot the average image separately
    fig = plt.figure()
    plt.imshow(average_image, cmap='hot')
    plt.title("Average Image")
    plt.axis('off')  # Hide axes

    image_path = helper_functions.get_next_image_number()
    #write image to md
    if Parameters.save_runs_to_md:
        plt.savefig(image_path, dpi=300)
        helper_functions.write_to_md("quick_view_images<p>",image_path)

    mlflow.log_figure(fig, image_path)

    #plt.show()
    plt.close(fig)


def plot_external_test_graphs(params, test_stack_input, train_stack_input,
                              image_series_correlations, image_series_mean_correlation):

    #scatter actual vs predicted
    scatter_diagram_twovar_plot_mean(params.external_test_stock_ticker,params.train_stock_ticker,test_stack_input, train_stack_input)

    #plot trained versus test stocks image series mean correlations
    plot_image_correlations(image_series_correlations, image_series_mean_correlation)
    print("trained versus test stocks image series mean correlation",image_series_mean_correlation)