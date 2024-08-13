import os
import sys
import torch
import numpy as np

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import helper_functions as helper_functions

def compute_and_report_error_stats(stack_actual, stack_predicted, stock_ticker):
    #compute stats
    error_stats = compute_error_stats(stack_actual, stack_predicted)
    text_mssg=f"Error Stats for {stock_ticker}<p>"
    helper_functions.write_to_md(text_mssg,None)
    print(f"Error Stats for {stock_ticker}")
    for key, value in error_stats.items():
        text_mssg=f'{key}: {value}<p>'
        helper_functions.write_to_md(text_mssg,None)
        print(f'{key}: {value}\n')

def compute_error_stats(var1, var2):
    mae = torch.mean(torch.abs(var1 - var2))

    mse = torch.mean((var1 - var2) ** 2)

    rmse = torch.sqrt(mse)

    mape = torch.mean(torch.abs((var1 - var2) / var1)) * 100

    ss_total = torch.sum((var1 - torch.mean(var1)) ** 2)
    ss_residual = torch.sum((var1 - var2) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    return {
        'MAE': mae.item(),
        'MSE': mse.item(),
        'RMSE': rmse.item(),
        'MAPE': mape.item(),
        'R2': r2.item()
    }

def self_correlation_feature_1_feature_2(stock_df,feature_1,feature_2):
    correlation = stock_df[feature_1].corr(stock_df[feature_2])
    return (f'Correlation between {feature_1} and {feature_2}: {correlation:.4f}')

def stock_correlation_matrix(stock_ticker,stock_df):
    correlation_matrix = stock_df.corr(method='pearson')
    print("Stock Correlation",stock_ticker)
    print(correlation_matrix)

def cross_stock_df_correlation(stock_ticker,index_ticker,stock_1_df, stock_2_df):
    #align frames
    stock_1_data, stock_2_data = stock_1_df.align(stock_2_df, join='inner')
    
    correlations = {}
    for column in stock_1_df.columns:
        correlation = stock_1_data[column].corr(stock_2_data[column])
        correlations[column] = correlation

    for feature, correlation in correlations.items():
        print(f'Price Correlation between {feature} of {stock_ticker} and {index_ticker}: {correlation:.4f}')


def cross_stock_image_array_correlation(var1, var2):
    var1 = var1.view(64, -1)
    var2 = var2.view(64, -1)
    
    mean1 = torch.mean(var1, dim=1, keepdim=True)
    mean2 = torch.mean(var2, dim=1, keepdim=True)
    means = torch.stack((mean1, mean2), dim=0)
    means = means.squeeze(-1) 

    print("var2.shape",var2.shape,"shape mean",mean2.shape, "means shape",means.shape)
    
    # Compute the correlation matrix
    correlation_matrix = torch.corrcoef(means)

    correlation_matrix_np = correlation_matrix.numpy()

    return correlation_matrix

def cross_stock_image_array_correlation2(var1, var2):
    reshaped_var1 = var1.view(var1.size(0), var1.size(1), -1)
    reshaped_var2 = var2.view(var1.size(0), var2.size(1), -1)
    
    #print("var1 shape",var1.shape,"reshaped",reshaped_var1.shape)

    correlations = []
    #loop through channels for each batch
    for batch_idx in range(reshaped_var1.size(0)):
    #loop through images of each channel
        for chann in range(reshaped_var1.size(1)):
            var1_chann_image = reshaped_var1[batch_idx, chann, :]
            var2_chann_image = reshaped_var2[batch_idx, chann, :]
            stacked_tensors = torch.stack([var1_chann_image, var2_chann_image])
            #print("var1 shape",var1_chann_image.shape,"var2 shape",var2_chann_image.shape,"stacked shape",stacked_tensors.shape)
            correlation_matrix = torch.corrcoef(stacked_tensors)
            correlation = correlation_matrix[0, 1].item()
            correlations.append(correlation)

    correlations = torch.tensor(correlations).numpy()
    mean_correlation = correlations.mean()
    
    return correlations, mean_correlation

def calculate_iqr(tensor_list):
    stack = torch.stack(tensor_list, dim=1)
    for stack_tensor in stack:
        #print("stack_tensor shape",stack_tensor.shape)
        Q1 = np.percentile(stack_tensor, 25)
        Q3 = np.percentile(stack_tensor, 75)
        IQR = Q3 - Q1
    
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        data_iqr = [x for x in stack_tensor if lower_bound <= x <= upper_bound]
        #print("dataiqr",data_iqr)
        error_pct_outside_iqr = ((len(stack_tensor) - len(data_iqr)) / len(stack_tensor)) * 100
        #print("len dataiqr",len(data_iqr))
        #print("dropped",percentage_dropped)

    return data_iqr, error_pct_outside_iqr