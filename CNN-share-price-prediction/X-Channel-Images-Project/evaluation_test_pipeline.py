import os
import sys
import mlflow

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import helper_functions_dir.load_data as load_data
import helper_functions_dir.compute_stats as compute_stats
import helper_functions_dir.helper_functions as helper_functions

# report stats results
def report_evaluation_test_stats(stocks, params, evaluation_test_stock_dataset_df,
                               train_stack_input, evaluation_test_stack_input,
                               run, experiment_name):
    
    
    #compute correl prices
    compute_stats.stock_correlation_matrix(params.eval_tickers, evaluation_test_stock_dataset_df)
    #compute correl images
    image_series_correlations, image_series_mean_correlation = compute_stats.cross_stock_image_array_correlation2(evaluation_test_stack_input, train_stack_input, params.eval_tickers, params.train_tickers)

    #compute cross correl
    # train_stock_df, stock_tickers = load_data.import_dataset(stocks, params.start_date, params.end_date, run, experiment_name)
    # compute_stats.cross_stock_df_correlation(params.eval_tickers, params.train_tickers,evaluation_test_stock_dataset_df, train_stock_df)
        
    if params.enable_mlflow:
        mlflow.log_metric(f"Correlation_Mean_Trained_{params.train_tickers}_vs_Test_{params.eval_tickers}_Input_Image",image_series_mean_correlation.item())
    
    return image_series_correlations, image_series_mean_correlation