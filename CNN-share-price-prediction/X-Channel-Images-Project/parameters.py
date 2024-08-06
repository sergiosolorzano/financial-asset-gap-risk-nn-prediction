from sklearn.preprocessing import StandardScaler,MinMaxScaler
import sys
import os

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions'))
import helper_functions.generate_images as generate_images

#init parameters
class Parameters:
    scenario = 0

    brute_force_filename = 'brute_force_results.md'

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
    sample_range = (0, 1)
    
    # GRAMIAN/MARKOV: image transformation scale
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = StandardScaler()#[StandardScaler(), MinMaxScaler()]

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

    loss_threshold = 0.0001

    epoch_running_loss_check = 500
    
    epoch_running_gradients_check = 4000
