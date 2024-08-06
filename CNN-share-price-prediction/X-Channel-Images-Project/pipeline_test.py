import os
import sys

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions'))
import helper_functions.neural_network as neural_network
import helper_functions.plot_data as plot_data
import helper_functions.compute_stats as compute_stats
import helper_functions.helper_functions as helper_functions

def test_process(net, test_loader, params, stock_ticker):
    
    # test
    stack_input, predicted_list, actual_list, accuracy, stack_actual, stack_predicted  = neural_network.Test(test_loader,net)

    # Plot image mean input values
    plot_data.scatter_diagram_onevar_plot_mean(stack_input, stock_ticker)
    
    #compute stats
    compute_stats.compute_and_report_error_stats(stack_actual, stack_predicted, stock_ticker)

    #write to file
    helper_functions.write_scenario_to_log_file(accuracy)

    return stack_input, stack_actual, stack_predicted