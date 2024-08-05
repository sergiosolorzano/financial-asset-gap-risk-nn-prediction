from __future__ import print_function

import os
import glob
import numpy as np

from enum import Enum

import torch
print(torch.__version__)

def Delete_Scenario_Files():
    scenario_files = glob.glob('*scenario*') #find files
    iteration_files = glob.glob('*iteration*')
    optimizer_files = glob.glob('*optimizer*')
    model_files = glob.glob('*model*')

    files_to_delete = scenario_files + iteration_files + optimizer_files + model_files
    files_to_delete = [file for file in files_to_delete if file != 'model_card.md']
    
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")

def Save_BayesOpt_Model(scenario, net):
    PATH = f'./model_bayesOpt_iteration_{scenario}.pth'
    torch.save(net.state_dict(), PATH)

def Save_Scenario_State_Model(scenario, net):
    PATH = f'./model_scen_{scenario}_state.pth'
    torch.save(net.state_dict(), PATH)

def Save_Scenario_Full_Model(scenario, net):
    PATH = f'./model_scen_{scenario}_full.pth'
    torch.save(net, PATH)

def Load_State_Model(net, PATH):
    print("Loading State Model")
    net.load_state_dict(torch.load(PATH))
    net.eval()
    return net

def Load_Full_Model(PATH):
    if os.path.exists(PATH):
        print("Loading Full Model")
        net = torch.load(PATH)
        net.eval()
        return net
    else:
        print(f"Model {PATH} does not exist")

def Clear_Scenario_Log(scenario):
    with open(f'scenario_{scenario}.txt', 'w') as file:
        file.write('')

def Scenario_Log(output_string):
    with open(f'scenarios_output.txt', 'a') as file:
        file.write('\n\n' + output_string)

def Load_BayesOpt_Model(scenario, net):
    PATH = f'./bayesian_optimization_saved_models/model_bayesOpt_iteration_{scenario}.pth'
    print("path model is ",PATH)
    net.load_state_dict(torch.load(PATH))
    net.eval()
    return net

def data_to_array(input_list):
    output_array = np.array(input_list)

    return output_array

def write_scenario_to_log_file(accuracy):
    #write to file
    output_string = (f"Accuracy 2dp: {accuracy[0]}%\n"
                    f"Accuracy 1dp: {accuracy[1]}%\n")

    Scenario_Log(output_string)

def Save_Model(scenario, net):
    Save_Scenario_State_Model(scenario,net)
    Save_Scenario_Full_Model(scenario,net)

