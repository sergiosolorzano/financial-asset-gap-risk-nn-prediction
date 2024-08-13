import os
import sys

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import helper_functions_dir.load_data as load_data
import helper_functions_dir.compute_stats as compute_stats
import helper_functions_dir.helper_functions as helper_functions

# report stats results
def report_external_test_stats(params, stock_dataset_df, 
                               test_stack_input, train_stack_input,
                               test_stack_actual, test_stack_predicted,
                               run):
    
    #test_stack_actual is the actual observation
    #test_stack_predicted is the predicted observation

    #print("test_stack_input.shape",test_stack_input.shape,"test_stack input",test_stack_input)
    #compute correl prices
    compute_stats.stock_correlation_matrix(params.external_test_stock_ticker, stock_dataset_df)
    #compute correl images
    #print("trained input image shape",train_stack_input.shape,"test input image shape",test_stack_input.shape)
    image_series_correlations, image_series_mean_correlation = compute_stats.cross_stock_image_array_correlation2(test_stack_input,train_stack_input)

    #compute cross correl
    benchmark_stock_df = load_data.import_dataset(params.train_stock_ticker, params.start_date, params.end_date, run)
    compute_stats.cross_stock_df_correlation(params.external_test_stock_ticker, params.train_stock_ticker,stock_dataset_df, benchmark_stock_df)

    # #compute stats
    # error_stats = compute_stats.compute_error_stats(test_stack_actual, test_stack_predicted)
    # text_mssg=(f"Inference Model Stats for {params.external_test_stock_ticker}<p>")
    # print(text_mssg)
    # for key, value in error_stats.items():
    #     text_mssg=(f'{key}: {value}<p>')
    #     print(text_mssg)
    #     helper_functions.write_to_md(text_mssg,None)
        
    return image_series_correlations, image_series_mean_correlation