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
import process_price_series as process_price_series

import torch
import load_data

matplotlib.use(Parameters.matplotlib_use)

def plot_weights_gradients(weights_dict, gradients_dict, epoch, experiment_name, run_id):
    for name, weight_list in weights_dict.items():
        fig = plt.figure(figsize=(8, 4))
        plt.title(f"Epoch {epoch + 1} - Weights {name}")
        
        for i, w in enumerate(weight_list):
            plt.plot(w.flatten(), label=f'Weight {i}')
        
        plt.xlabel('Weight Index')
        plt.ylabel('Weight Value')
        plt.legend(loc="upper right")

        md_name = "plot_weights<p>"
        helper_functions.write_and_log_plt(fig, epoch+1, name, md_name, experiment_name, run_id)

    for name, gradient_list in gradients_dict.items():
        fig = plt.figure(figsize=(10, 6))
        plt.title(f"Epoch {epoch + 1} - Gradients {name}")
        
        for i, g in enumerate(gradient_list):
            plt.plot(g.flatten(), label=f'Gradient {i}')
        
        plt.xlabel('Gradient Index')
        plt.ylabel('Gradient Value')
        plt.legend(loc="upper right")

        #write image to md
        # if Parameters.save_runs_to_md:
        #image_path = helper_functions.get_next_image_number()        
        #     plt.savefig(image_path, dpi=300)
        #     helper_functions.write_to_md("plot_gradients<p>",image_path)

        md_name = "plot_gradients<p>"
        helper_functions.write_and_log_plt(fig, epoch+1, name, md_name, experiment_name, run_id)

        #plt.show()
        plt.close(fig)

def scatter_diagram_onevar_plot_mean(stack_input, stock_ticker, experiment_name, run_id):
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

    helper_functions.write_and_log_plt(fig, None, f'{stock_ticker}_Image_Input_Mean_Values',
                                       f'{stock_ticker}_Image_Input_Mean_Values', experiment_name, run_id)
    #plt.show()
    plt.close(fig)

def scatter_diagram_twovar_plot_mean(evaluation_test_stock_ticker,train_stock_ticker,var1, var2, experiment_name, run_id):
    torch.set_printoptions(threshold=torch.inf)
    reshaped_evaluation_test_stack_input = var1.view(var1.size(0), var1.size(1), -1)
    reshaped_train_stock_ticker = var2.view(var2.size(0), var2.size(1), -1)
    
    print("Scatter len eval set",len(reshaped_evaluation_test_stack_input), "shape",reshaped_evaluation_test_stack_input.shape,"len train set",len(reshaped_train_stock_ticker),"shape",reshaped_train_stock_ticker.shape)
    
    test_means =[]
    train_means =[]
    for batch_idx in range(reshaped_evaluation_test_stack_input.size(0)):
    #loop through images of each channel
        for chann in range(reshaped_evaluation_test_stack_input.size(1)):
            var1_chann_image = reshaped_evaluation_test_stack_input[batch_idx, chann, :]
            var2_chann_image = reshaped_train_stock_ticker[batch_idx, chann, :]
            var1_mean_value = torch.mean(var1_chann_image).item()
            var2_mean_value = torch.mean(var2_chann_image).item()
            test_means.append(var1_mean_value)
            train_means.append(var2_mean_value)

    fig = plt.figure(figsize=(10, 6))
    df = pd.DataFrame({
        f'{evaluation_test_stock_ticker} test_mean': test_means,
        f'{train_stock_ticker} train_mean': train_means
    })
    #print("input mean",df)
    plt.scatter(df.index, df[f'{evaluation_test_stock_ticker} test_mean'], c='red', marker='x', label=f'{evaluation_test_stock_ticker} Mean Values')
    plt.scatter(df.index, df[f'{train_stock_ticker} train_mean'], c='blue', marker='x', label=f'{train_stock_ticker} Mean Values')
    plt.xlabel(f'{train_stock_ticker} and {evaluation_test_stock_ticker} Input Mean Values')
    plt.ylabel('Values')
    plt.title(f'Scatter Diagram of {evaluation_test_stock_ticker} and {train_stock_ticker} Image Input Mean Values')
    plt.legend(loc="upper right")
    plt.grid(True)

    helper_functions.write_and_log_plt(fig, None,
                                       f"{evaluation_test_stock_ticker}_and_{train_stock_ticker}_Image_Input_Mean_Values",
                                       f"{evaluation_test_stock_ticker}_and_{train_stock_ticker}_Image_Input_Mean_Values", experiment_name, run_id)
    
def plot_confusion_matrix(conf_matrix, stock_ticker, normalize, experiment_name, run_id):
    raw_conf_matrix = conf_matrix.copy()
    
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix = np.round(conf_matrix * 100, 2)

    # Create annotations with both raw counts and percentages
    annotations = np.empty(conf_matrix.shape, dtype=object)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            count = raw_conf_matrix[i, j]
            percent = conf_matrix[i, j] if normalize else 0
            annotations[i, j] = f'{count}\n({percent}%)'

    if normalize:
        fmt='0.2f'
    else:
        fmt='d'

    # Plot confusion matrix
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=['Next Day Price Down', 'Next Day Price Up'], 
                yticklabels=['Next Day Price Down', 'Next Day Price Up'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix Test {stock_ticker}')

    if Parameters.enable_mlflow:
        helper_functions.write_and_log_plt(fig, None, f'{stock_ticker}_Confusion_Matrix', f'{stock_ticker}_Confusion_Matrix', experiment_name, run_id)
    #plt.show()
    plt.close(fig)

def quick_view_images(images_array, cols_used_count, cols_used, experiment_name, run_id):
    
    # Plot the first image of each column
    fig, axes = plt.subplots(nrows=1, ncols=cols_used_count, figsize=(20, 6))
    for ax in axes:
        ax.set_aspect('equal')

    #print("shape images array",images_array.shape,"shape image",images_array[0][0][0][0].shape)
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

    helper_functions.write_and_log_plt(fig, None,
                                       f"quick_view_image",
                                       f"quick_view_image", experiment_name, run_id)

    #plt.show()
    plt.close(fig)

def plot_evaluation_test_graphs(params, train_stack_input, evaluation_test_stack_input,
                              image_series_correlations, image_series_mean_correlation,
                              experiment_name, run_id):

    #scatter actual vs predicted
    scatter_diagram_twovar_plot_mean(params.eval_tickers,params.train_tickers,evaluation_test_stack_input, train_stack_input, experiment_name, run_id)

    #plot trained versus test stocks image series mean correlations
    print("len train_stack_input",len(train_stack_input),"len evaluation_test_stack_input",len(evaluation_test_stack_input))
    if len(train_stack_input) == len(evaluation_test_stack_input):
        plot_image_correlations(image_series_correlations, image_series_mean_correlation, experiment_name, run_id)
        print("trained versus test stocks image series mean correlation",image_series_mean_correlation)
    else:
        print(f"Skip Plot Cross-Image Correlations because of size mismatch: Train size {len(train_stack_input)} Eval size {len(evaluation_test_stack_input)}")

def plot_image_correlations(series_correlations, mean_correlation, experiment_name, run_id):
    # Plot the correlations
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(series_correlations, kde=True, bins=30)
    plt.axvline(mean_correlation, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_correlation:.2f}')
    plt.title('Distribution of Trained vs Test Stocks Input Image Series Correlations')
    plt.xlabel('Image Correlation Coefficient')
    plt.ylabel('Frequency')

    helper_functions.write_and_log_plt(fig, None,
                                       f"Correlation_Trained_vs_Test_Stocks_Input_Image_Series",
                                       f"Correlation_Trained_vs_Test_Stocks_Input_Image_Series", experiment_name, run_id)

    #plt.show()
    plt.close(fig)

def plot_price_comparison_stocks(index_ticker, train_stock_tickers, stock_dataset_df, stocks):
    fig = compare_stocks(index_ticker,train_stock_tickers,stock_dataset_df, stocks)
    
    return fig

def plot_concat_price_comparison_stocks(train_stock_tickers, stock_dataset_df):
    fig = compare_concat_stocks(train_stock_tickers,stock_dataset_df)
    
    return fig

def plot_merged_log_series(merged_df, experiment_name, run_id):
    fig = plt.figure(figsize=(10, 6))
    
    # Loop through the columns (tickers) except 'Date'
    for column in merged_df.columns:
        if column != 'Date':
            plt.plot(merged_df['Date'], merged_df[column], label=column)

    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel('Rebased Price')
    plt.title(f'Log Rebased Time Series for {[col for col in merged_df.columns if col != 'Date']}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    helper_functions.write_and_log_plt(fig, None,
                                       f"Log_Rebased_Time_Series_for_{[col for col in merged_df.columns if col != 'Date']}",
                                       f"Log_Rebased_Time_Series_for_{[col for col in merged_df.columns if col != 'Date']}", experiment_name, run_id)

    #plt.show()
    plt.close(fig)

def compare_concat_stocks(stock_ticker, stock_dataset):

    stock_data = stock_dataset.dropna()

    #rebase
    stock_rebased = stock_data# / stock_data.iloc[0] * 100

    fig, axs = plt.subplots(2, 2, figsize=(Parameters.plt_image_size[0], Parameters.plt_image_size[1]))

    axs[0, 0].plot(stock_rebased.index, stock_rebased['Open'], label=f'{stock_ticker} Open Price', color='g')
    axs[0, 0].plot(stock_rebased.index, stock_rebased['Close'], label=f'{stock_ticker} Close Price', color='m')
    axs[0, 0].set_title('Open and Close Prices')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(stock_rebased.index, stock_rebased['Close'], label=f'{stock_ticker} Close Price', color='m')
    axs[0, 1].plot(stock_rebased.index, stock_rebased['High'], label=f'{stock_ticker} High Price', color='b')
    axs[0, 1].set_title('Close and High Prices')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(stock_rebased.index, stock_rebased['Close'], label=f'{stock_ticker} Close Price', color='m')
    axs[1, 0].plot(stock_rebased.index, stock_rebased['Low'], label=f'{stock_ticker} Low Price', color='r')
    axs[1, 0].set_title('Close and Low Prices')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(stock_rebased.index, stock_rebased['High'], label=f'{stock_ticker} High Price', color='b')
    axs[1, 1].plot(stock_rebased.index, stock_rebased['Low'], label=f'{stock_ticker} Low Price', color='r')
    axs[1, 1].set_title('High and Low Prices')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    #plt.show()

    return fig

def compare_stocks(index_ticker, stock_ticker, stock_dataset, stocks):

    start_date, end_date = stocks.get_dates_by_ticker(stock_ticker)

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
    #plt.show()

    return fig

def plot_dtw_matrix(distance_matrix, stock_tickers, experiment_name, run_id):
    print("AT plot_dtw_matrix",distance_matrix)
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(distance_matrix.astype(float), annot=True, cmap='coolwarm', fmt='.2f', xticklabels=stock_tickers, yticklabels=stock_tickers)
    plt.title('DTW Distance Heatmap')
    plt.xlabel('Stock Tickers')
    plt.ylabel('Stock Tickers')
    print("Call Image DTW Map")
    helper_functions.write_and_log_plt(fig, None,
                                       f"DTW Distance Heatmap",
                                       f"DTW Distance Heatmap", experiment_name, run_id)

    #plt.show()
    plt.close(fig)
    
def plot_train_series_correl(cross_corr_matrix, experiment_name, run_id):
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(cross_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Cross-Correlation Matrix Time Series')

    helper_functions.write_and_log_plt(fig, None,
                                       f"Cross_Correlation_Matrix_Time_Series",
                                       f"Cross_Correlation_Matrix_Close_Time_Series", experiment_name, run_id)

    #plt.show()
    plt.close(fig)
    
def plot_all_cross_correl_price_series(stocks, run, experiment_name):
    
    data_close, merged_df = process_price_series.log_rebase_dataset(stocks)
    
    df_close = pd.DataFrame(data_close)
    cross_corr_matrix = df_close.corr(method='spearman')
    
    if Parameters.enable_mlflow:
        plot_train_series_correl(cross_corr_matrix, experiment_name, run.info.run_id)

# def plot_train_eval_cross_correl_price_series(stocks, run, experiment_name):
#     data_close = {}

#     for s in stocks.get_train_stocks() + stocks.get_eval_stocks():
#         dataset_df = yf.download(s['ticker'], start=s['start_date'], end=s['end_date'], interval='1d')
#         dataset_df = dataset_df.dropna()
#         #reset column to save to csv and mlflow schema
#         dataset_df = dataset_df.reset_index()

#         #reorder to split the data to train and test
#         desired_order = ['Date','Open', 'Close', 'High', 'Low']
#         if 'Date' in dataset_df.columns:
#             dataset_df = dataset_df[desired_order]
#         else:
#             print("Column 'Date' is missing.")

#         data_close[s['ticker']] = dataset_df['Close']

#         # calc correl training datasets
#         df_close = pd.DataFrame(data_close)
        
#     # print("Cross Correl for tickers ",data_close.keys)
#     # print("cross correl data",df_close)
#     cross_corr_matrix = df_close.corr(method='spearman')
#     #print("Train & Eval set cross_corr_matrix",cross_corr_matrix)
#     # from fastdtw import fastdtw
#     # distance, path = fastdtw(data_close['SIVBQ'], data_close['CMA'])
#     # print("**distance",distance,"path", path)
    
#     if Parameters.enable_mlflow:
#         plot_train_series_correl(cross_corr_matrix, experiment_name, run.info.run_id)

def dtw_map_all(stocks, run, experiment_name):
    
    data_close, merged_df = process_price_series.log_rebase_dataset(stocks)

    print("DTW for tickers ",data_close.keys())
    stock_tickers = stocks.get_all_tickers()
    
    distance_matrix = pd.DataFrame(index=stock_tickers, columns=stock_tickers)
    for i, stock_ticker in enumerate(stock_tickers):
        if stock_ticker in data_close:
            for j, compare_ticker in enumerate(stock_tickers):
                if compare_ticker in data_close:
                    distance, path = fastdtw(data_close[stock_ticker], data_close[compare_ticker])
                    distance_matrix.iloc[i, j] = distance
                    #print(f"**distance {stock_ticker} vs {compare_ticker}: {distance}, path: {path}")
                    # if Parameters.enable_mlflow:
                    #     mlflow.log_param(f"dwt_path_train_{stock_ticker}_eval_{eval_ticker}",path)

    plot_dtw_matrix(distance_matrix,stock_tickers,experiment_name,run.info.run_id)



