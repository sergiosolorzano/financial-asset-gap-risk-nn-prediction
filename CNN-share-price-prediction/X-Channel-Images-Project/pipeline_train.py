import os
import sys

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions'))
import helper_functions.neural_network as neural_network
import helper_functions.helper_functions as helper_functions

def train_process(train_loader, params):
    #init cnn
    net = neural_network.instantiate_net(params)

    # train cnn
    net = neural_network.Train(params, train_loader, net)

    helper_functions.Save_Model(params.scenario, net)

    return net