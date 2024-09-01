import os
import sys
import torch
import numpy as np
from torcheval.metrics import R2Score
#from torcheval.metrics.functional.regression.r2_score import _r2_score_compute

import mlflow
from parameters import Parameters

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import helper_functions as helper_functions

def compute_and_report_error_stats(stack_actual, stack_predicted, stock_ticker, device):
    #compute stats
    error_stats = compute_error_stats(stack_actual, stack_predicted, stock_ticker, device)
    
    if Parameters.save_runs_to_md:
        text_mssg=f"Error Stats for {stock_ticker}<p>"
        print(f"Error Stats for {stock_ticker}")
        helper_functions.write_to_md(text_mssg,None)
        for key, value in error_stats.items():
            text_mssg=f'{key}: {value}<p>'
            helper_functions.write_to_md(text_mssg,None)
            print(f'{key}: {value}\n')

def compute_error_stats(var1, var2, stock_ticker, device):
    # print("**shape var1",var1.shape,"var1[0].shape",var1[0].shape,"var 1 len",len(var1))
    # print("**shape var1",var2.shape,"var2[0].shape",var2[0].shape,"var 1 len",len(var2))
    # print("var1",var1)
    # print("var2",var2)
    mae = torch.mean(torch.abs(var1 - var2))

    mse = torch.mean((var1 - var2) ** 2)

    rmse = torch.sqrt(mse)

    mape = torch.mean(torch.abs((var1 - var2) / var1)) * 100

    #R^2: Currently being investigated manual vs torcheval
    ss_total = torch.sum((var1 - torch.mean(var1)) ** 2)
    ss_residual = torch.sum((var1 - var2) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    print("R^2 manual",r2, "my ss_total", ss_total, "ss_residual", ss_residual)

    metric = R2Score(device=device)
    update = metric.update(var1, var2)
    print("sum_squared_residual",update.sum_squared_residual)
    print("sum_obs",update.sum_obs)
    print("torch.square(sum_obs)",torch.square(update.sum_obs))
    print("num_obs",len(var1))
    print("sum_squared_obs",update.sum_squared_obs)
    r2_py = metric.compute()
    print("R^2 pytorch",r2_py)

    mae_cpu = mae.double().item()
    mse_cpu = mse.double().item()
    rmse_cpu = rmse.double().item()
    mape_cpu = mape.double().item()
    r2_cpu = r2.double().item()
    r2_py_cpu = r2_py.double().item()

    error_metrics = {f"{stock_ticker} MAE": mae_cpu,
               f"{stock_ticker} MSE": mse_cpu,
               f"{stock_ticker} RMSE": rmse_cpu,
               f"{stock_ticker} R2": r2_cpu,
               f"{stock_ticker} R2_py": r2_py_cpu
               }
    
    mlflow.log_metrics(error_metrics)

    return {
        'MAE': mae_cpu,
        'MSE': mse_cpu,
        'RMSE': rmse_cpu,
        'MAPE': mape_cpu,
        'R2': r2_cpu,
        'R2_py': r2_py_cpu
    }

def self_correlation_feature_1_feature_2(stock_df,feature_1,feature_2):
    correlation = stock_df[feature_1].corr(stock_df[feature_2])

    mlflow.log_metrics(f"correlation_{feature_1}_vs_{feature_2}", correlation)

    return (f'Correlation between {feature_1} and {feature_2}: {correlation:.4f}')

def stock_correlation_matrix(stock_ticker,stock_df):
    correlation_matrix = stock_df.corr(method='pearson')
    print("Stock X-Correlation",stock_ticker)
    print(correlation_matrix)


def cross_stock_df_correlation(stock_ticker,index_ticker,stock_1_df, stock_2_df):
    #align frames
    stock_1_data, stock_2_data = stock_1_df.align(stock_2_df, join='inner')
    
    correlations = {}
    price_correl_metrics = {}
    for column in stock_1_df.columns:
        #print("Calc Correl column", column)
        correlation = stock_1_data[column].corr(stock_2_data[column])
        correlations[column] = correlation

    for feature, correlation in correlations.items():
        #print(f'Price Correlation between {feature} of {stock_ticker} and {index_ticker}: {correlation:.4f}')
        price_correl_metrics[f"Correl_Prices_feature_{feature}_stock_{stock_ticker}_vs_{index_ticker}"] = correlation
    
    mlflow.log_metrics(price_correl_metrics)
    


def cross_stock_image_array_correlation(var1, var2):
    var1 = var1.view(64, -1)
    var2 = var2.view(64, -1)
    
    mean1 = torch.mean(var1, dim=1, keepdim=True)
    mean2 = torch.mean(var2, dim=1, keepdim=True)
    means = torch.stack((mean1, mean2), dim=0)
    means = means.squeeze(-1) 
    
    # Compute the correlation matrix
    correlation_matrix = torch.corrcoef(means)

    return correlation_matrix

def cross_stock_image_array_correlation2(var1, var2, test_ticker, train_ticker):
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
    stack_flat_np = torch.stack(tensor_list, dim=0).detach().cpu().numpy()
    
    Q1 = np.percentile(stack_flat_np, 25)
    Q3 = np.percentile(stack_flat_np, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    data_iqr_np = stack_flat_np[(stack_flat_np >= lower_bound) & (stack_flat_np <= upper_bound)]
    error_pct_outside_iqr = ((len(stack_flat_np) - len(data_iqr_np)) / len(stack_flat_np)) * 100

    return data_iqr_np, error_pct_outside_iqr