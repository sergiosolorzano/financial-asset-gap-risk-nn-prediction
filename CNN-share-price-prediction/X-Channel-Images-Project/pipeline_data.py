import os
import sys
import pandas as pd
import numpy as np
import mlflow

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..","..","..")))
import helper_functions_dir.load_data as load_data
import helper_functions_dir.plot_data as plot_data
import helper_functions_dir.image_transform as image_transform
from helper_functions_dir import helper_functions
#from parameters import Parameters

#import mlflow

def remap_to_log_returns(stocks_dataset_df, start_indices_cumulative):
    np.set_printoptions(threshold=np.inf)
    pd.set_option('display.max_rows', None)

    #convert to log returns
    #print("REMAP stock_log_returns_df",stocks_dataset_df)
    #print("cum index",start_indices_cumulative, "len df", len(stocks_dataset_df))
    stock_log_returns_df = pd.DataFrame()
    numeric_cols = stocks_dataset_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        stock_log_returns_df[col] = np.log(stocks_dataset_df[col] / stocks_dataset_df[col].shift(1))
    
    # set the log return concat location for the series to zero
    if start_indices_cumulative is not None and len(start_indices_cumulative) > 0:
        for counter, i in enumerate(start_indices_cumulative):
            stock_log_returns_df.iloc[i-counter-1]=0
    
    stock_log_returns_df = stock_log_returns_df.dropna()
    #print("log stock_log_returns_df",stock_log_returns_df)

    #rebase
    rebased_df = pd.DataFrame()
    for col in stock_log_returns_df.columns:
        rebased_values = [100]
        for log_return in stock_log_returns_df[col]:
            next_rebased_value = rebased_values[-1] * (1 + log_return)
            rebased_values.append(next_rebased_value)
        rebased_df[col] = rebased_values[:-1] 
        
    # pd.set_option('display.max_rows', None)
    # print("log rebased df",rebased_df)

    return rebased_df

def generate_dataset_to_images_process(stocksobj, stocks, params, test_size, cols_used, run, experiment_name, stock_type):

    stocks_dataset_df, start_indices_cumulative, stock_tickers = load_data.import_dataset(stocks, run, experiment_name)
    
    if params.log_returns:
        #print("Raw Dataset",stocks_dataset_df)
        log_rebased_df = remap_to_log_returns(stocks_dataset_df, start_indices_cumulative)
        
        print("Rebased DF to GENERATE IMAEG",stock_tickers,"log_rebased_df",log_rebased_df)
        
        stocks_dataset_df = log_rebased_df
        print(f"LEN log stocks_dataset_df {stock_tickers}",len(stocks_dataset_df))
        if stock_type=="eval":
            if params.enable_mlflow:
                mlflow.log_params({"eval_daycount": len(stocks_dataset_df)})
        if stock_type=="train":
            if params.enable_mlflow:
                mlflow.log_params({"train_daycount": len(stocks_dataset_df)})
        

    # Generate images
    #print("generate_dataset_to_images_process algo",params.transform_algo)
    
    feature_image_dataset_list, feature_price_dataset_list, feature_label_dataset_list, cols_used_count = image_transform.generate_features_lists(
        stocks_dataset_df, 
        cols_used,
        params.transform_algo, 
        params.transformed_img_sz, 
        params.window_overlap,
        params.window_method,
        params.gaf_method, 
        params.gaf_sample_range)

    #print("LABEL LIST",feature_label_dataset_list)
    images_array, labels_array = image_transform.create_images_array(feature_image_dataset_list, feature_label_dataset_list)
    print("***image shape",images_array.shape)
    
    #Quick Sample Image Visualization
    #Visualize Closing Price for one image in GAF or Markov:
    # A darker patch indicates lower correlation between the different elements of the price time series, 
    # possibly due to higher volatility or noise. The opposite is true for the lighter patches.
    plot_data.quick_view_images(images_array, cols_used_count, cols_used, experiment_name, getattr(run, 'info', None).run_id if run else None)

    #Prepare and Load Data
    images_array, labels_array = image_transform.squeeze_array(images_array, labels_array)
    
    feature_image_dataset_list_f32, labels_scaled_list_f32 = image_transform.Generate_feature_image_to_f32(
        labels_array, 
        images_array,
        params.transformed_img_sz, 
        params.scaler)

    #print("***---***labels",labels_scaled_list_f32)
    train_loader, test_loader = load_data.Generate_Loaders(feature_image_dataset_list_f32,
                                                labels_scaled_list_f32, test_size,
                                                params.batch_size,
                                                train_shuffle=False)
    
    return train_loader, test_loader, stocks_dataset_df, feature_image_dataset_list_f32