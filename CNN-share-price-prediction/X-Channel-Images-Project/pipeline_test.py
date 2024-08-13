import os
import sys
import pandas as pd
import torch
import numpy as np

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import helper_functions_dir.neural_network as neural_network
import helper_functions_dir.plot_data as plot_data
import helper_functions_dir.compute_stats as compute_stats
import helper_functions_dir.helper_functions as helper_functions
from parameters import Parameters

def test_process(net, test_loader, params, stock_ticker, run):
    
    # test
    stack_input, predicted_list, actual_list, accuracy, stack_actual, stack_predicted  = neural_network.Test(test_loader,net)

    #store and to mlflow these input and predicted stacks
    reshaped_input_stack_tensor = stack_input.view(-1, stack_input.size(-1))
    input_df = pd.DataFrame(reshaped_input_stack_tensor.numpy())
    blob_name = f"{Parameters.input_image_data_blob_fname}_{run.info.run_id}.csv"
    full_blob_uri = helper_functions.save_df_to_blob(input_df, blob_name)
    tags = {'original_shape': f'{stack_input.shape}'}
    helper_functions.mlflow_log_dataset(input_df, full_blob_uri, stock_ticker, "input_image", "train", run, tags)
    
    reshaped_predicted_stack_tensor = stack_predicted.view(-1, stack_predicted.size(-1))
    predicted_df = pd.DataFrame(reshaped_predicted_stack_tensor.numpy())
    blob_name = f"{Parameters.predicted_image_data_blob_fname}_{run.info.run_id}.csv"
    full_blob_uri = helper_functions.save_df_to_blob(predicted_df, blob_name)
    tags = {'original_shape': f'{stack_predicted.shape}'}
    helper_functions.mlflow_log_dataset(predicted_df, full_blob_uri, stock_ticker, "predicted_image", "train", run, tags)

    #TODO:reconstruct to stack
    # numpy_array = df.to_numpy()
    # reconstructed_tensor = torch.tensor(numpy_array)
    # reconstructed_tensor = reconstructed_tensor.view(28, 16, 1, 32, 32)
    # print("Reconstructed Tensor Shape:", reconstructed_tensor.shape)
    
    # Plot image mean input values
    plot_data.scatter_diagram_onevar_plot_mean(stack_input, stock_ticker)
    
    #compute stats
    compute_stats.compute_and_report_error_stats(stack_actual, stack_predicted, stock_ticker)

    #write to file
    helper_functions.write_scenario_to_log_file(accuracy)

    return stack_input, stack_actual, stack_predicted