import os
import sys

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import helper_functions_dir.neural_network_enhanced as neural_network
import helper_functions_dir.helper_functions as helper_functions

def train_process(train_loader, train_feature_image_dataset_list_f32, train_stocks_dataset_df, params, run, run_id, experiment_name, device, stock_params):
    #init cnn
    net = neural_network.instantiate_net(params, device)

    # train cnn
    net, model_signature, train_stack_input, feature_maps_cnn_list, feature_maps_fc_list = neural_network.Train(params, train_loader, train_feature_image_dataset_list_f32, train_stocks_dataset_df, net, run, run_id, experiment_name, device, stock_params)

    helper_functions.Save_Model(run_id, net, model_signature, experiment_name, stock_params)

    return net, train_stack_input, feature_maps_cnn_list, feature_maps_fc_list