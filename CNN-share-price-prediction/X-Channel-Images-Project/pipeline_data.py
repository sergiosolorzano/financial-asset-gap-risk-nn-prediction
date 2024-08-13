import os
import sys

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..","..","..")))
import helper_functions_dir.load_data as load_data
import helper_functions_dir.plot_data as plot_data
import helper_functions_dir.image_transform as image_transform

import mlflow

def generate_dataset_to_images_process(stock_ticker, params, test_size, cols_used, run):
    #import Financial Data
    stock_dataset_df = load_data.import_dataset(stock_ticker, params.start_date, params.end_date, run)

    # plot price comparison stock vs index
    fig, image_path = plot_data.plot_price_comparison_stocks(params.index_ticker, stock_ticker, stock_dataset_df, params.start_date, params.end_date)
    mlflow.log_figure(fig, image_path)

    # Generate images
    print("generate_dataset_to_images_process algo",params.transform_algo)
    feature_image_dataset_list, feature_price_dataset_list, feature_label_dataset_list, cols_used_count = image_transform.generate_features_lists(
        stock_dataset_df, 
        cols_used,
        params.transform_algo, 
        params.transformed_img_sz, 
        params.gaf_method, 
        params.sample_range)

    images_array, labels_array = image_transform.create_images_array(feature_image_dataset_list, feature_label_dataset_list)

    #Quick Sample Image Visualization
    #Visualize Closing Price for one image in GAF or Markov:
    # A darker patch indicates lower correlation between the different elements of the price time series, 
    # possibly due to higher volatility or noise. The opposite is true for the lighter patches.
    if params.scenario == 0: plot_data.quick_view_images(images_array, cols_used_count, cols_used)

    #Prepare and Load Data
    images_array, labels_array = image_transform.squeeze_array(images_array, labels_array)

    feature_image_dataset_list_f32, labels_scaled_list_f32 = image_transform.Generate_feature_image_to_f32(
        labels_array, 
        images_array,
        params.transformed_img_sz, 
        params.scaler)

    train_loader, test_loader = load_data.Generate_Loaders(feature_image_dataset_list_f32,
                                                labels_scaled_list_f32, test_size,
                                                params.batch_size,
                                                train_shuffle=False)
    
    
    return train_loader, test_loader, stock_dataset_df