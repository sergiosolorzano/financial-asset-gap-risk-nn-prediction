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
def report_external_test_stats(params, external_test_stock_dataset_df,
                               train_stack_input, external_test_stack_input,
                               run):
    
    #compute correl prices
    compute_stats.stock_correlation_matrix(params.external_test_stock_ticker, external_test_stock_dataset_df)
    #compute correl images
    image_series_correlations, image_series_mean_correlation = compute_stats.cross_stock_image_array_correlation2(external_test_stack_input, train_stack_input, params.external_test_stock_ticker, params.train_stock_ticker)

    #compute cross correl
    train_stock_df = load_data.import_dataset(params.train_stock_ticker, params.start_date, params.end_date, run)
    compute_stats.cross_stock_df_correlation(params.external_test_stock_ticker, params.train_stock_ticker,external_test_stock_dataset_df, train_stock_df)
        
    mlflow.log_metric(f"Correlation_Mean_Trained_{params.train_stock_ticker}_vs_Test_{params.external_test_stock_ticker}_Input_Image",image_series_mean_correlation.item())
    
    return image_series_correlations, image_series_mean_correlation