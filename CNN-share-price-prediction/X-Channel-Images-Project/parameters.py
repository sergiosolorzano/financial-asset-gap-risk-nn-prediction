from sklearn.preprocessing import StandardScaler,MinMaxScaler
import sys
import os
import pandas as pd
from datetime import datetime

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
from helper_functions_dir import generate_images

#init parameters
class Parameters:
    scenario = 0

    mlflow_experiment_name = 'gaprisk-experiment-003'
    
    brute_force_filename = 'brute_force_results.md'
    mlflow_credentials_fname = 'mlflow-creds.json'
    input_price_data_blob_fname = 'input_price_data_run_id'
    input_image_data_blob_fname = 'input_image_data_run_id'
    predicted_image_data_blob_fname = 'predicted_image_data_run_id'

    save_runs_to_md = False

    save_arch_bool = True #only once

    plt_image_size = (12,5)
    
    brute_force_image_mlflow_dir = 'brute_force_images_mlflow'
    checkpoint_dir = 'model_checkpoints'
    full_model_dir = 'full_models'
    model_arch_dir = 'architecture_models'

    # Stock tickers
    train_stock_ticker = 'SIVBQ'
    external_test_stock_ticker = 'SICP'
    #test_stock_ticker = 'MSFT'
    index_ticker = '^SP500-40'
    
    # Close price time period
    start_date = '2021-12-05'
    end_date = '2023-01-25'

    #cols used
    training_cols_used = ["Open", "High", "Low", "Close"]
    external_test_cols_used = ["Open", "High"]

    # Time series to image transformation algorithm: GRAMIAN 1; MARKOV 2
    transform_algo_type = 1
    transform_algo = generate_images.TransformAlgo.from_value(transform_algo_type)
    image_resolution_x = 32
    image_resolution_y = 32
    
    # GAF image inputs
    gaf_method = "summation"
    transformed_img_sz = 32
    gaf_sample_range = (0, 1)
    
    # GRAMIAN/MARKOV: image transformation scale
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = StandardScaler()

    # Training's test size
    training_test_size = 0.5
    external_test_size = 1

    model_name ='LeNet-5 Based Net'
    # Default hyperparameters
    filter_size_1 = (2, 3)
    filter_size_2 = (2, 2)
    filter_size_3 = (2, 3)

    stride_1 = 1
    stride_2 = 2

    output_conv_1 = 40
    output_conv_2 = 12
    output_FC_1 = 100
    output_FC_2 = 70
    final_FCLayer_outputs = 1

    learning_rate = 0.00001
    momentum = 0.9

    dropout_probab = 0

    batch_size = 16

    num_epochs_input = 10000

    best_checkpoint_cum_loss = 0.002

    loss_threshold = 0.0001

    max_stale_loss_epochs = 2000

    epoch_running_loss_check = 2500
    
    epoch_running_gradients_check = 4500

    checkpt_dict = {
            'run_id': None,
            'epoch': None,
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'loss': None,
            }
