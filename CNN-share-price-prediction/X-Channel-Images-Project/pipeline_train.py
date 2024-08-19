import os
import sys

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import helper_functions_dir.neural_network as neural_network
import helper_functions_dir.helper_functions as helper_functions

def train_process(train_loader, params, run_id, experiment_name):
    #init cnn
    net = neural_network.instantiate_net(params)

    # train cnn
    net, model_signature, train_stack_input = neural_network.Train(params, train_loader, net, run_id, experiment_name)

    helper_functions.Save_Model(run_id, net, model_signature, experiment_name)

    return net, train_stack_input