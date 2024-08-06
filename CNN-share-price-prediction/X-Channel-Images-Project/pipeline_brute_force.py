#!/usr/bin/env python

import os
import sys
from sklearn.preprocessing import StandardScaler,MinMaxScaler

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions'))
import helper_functions.neural_network as neural_network
import helper_functions.plot_data as plot_data
import helper_functions.helper_functions as helper_functions
import helper_functions.compute_stats as compute_stats 

from parameters import Parameters
import pipeline_data as pipeline_data
import pipeline_train as pipeline_train
import pipeline_test as pipeline_test
import external_test_pipeline as external_test_pipeline


def brute_force_function():
    
    transform_algo_types = [1,2]
    gaf_methods = ["summation","difference"]
    sample_ranges = [(-1, 0), (0, 0.5), (0.5, 1), (1, 1)]
    scalers = [StandardScaler(), MinMaxScaler()]
    dropout_probabs = [0, 0.5, 0.8]

    for t in transform_algo_types:
        for m in gaf_methods:
            for s in sample_ranges:
                for sc in scalers:
                    for d in dropout_probabs:

                        Parameters.transform_algo_type = t
                        Parameters.gaf_method = m
                        Parameters.sample_range = s
                        Parameters.scaler = sc
                        Parameters.dropout_probab = d

                        helper_functions.write_to_md("==========<p>Optimization Iteration\==========<p>",None)
                        helper_functions.write_to_md(f"<p>transform_algo_type: {t} gaf_method: {m} sample_range: {s} scaler: {sc} dropout_probab: {d}<p>", None)
                        
                        #################################
                        #       Train and Test          #
                        #################################

                        #generate training images
                        train_loader, test_loader, stock_dataset_df = pipeline_data.generate_dataset_to_images_process(Parameters.train_stock_ticker, 
                                                                                    Parameters, 
                                                                                    Parameters.training_test_size, 
                                                                                    Parameters.training_cols_used)

                        net = pipeline_train.train_process(train_loader, Parameters)

                        #test
                        # set model to eval
                        net  = neural_network.set_model_for_eval(net)

                        test_stack_input, test_stack_actual, test_stack_predicted = pipeline_test.test_process(net, test_loader, 
                                                                                                Parameters, 
                                                                                                Parameters.train_stock_ticker)

                        #################################
                        #       External Test           #
                        #################################
                        text_mssg= "Run External Stock Tests:<p>"
                        print("\n\n",text_mssg)
                        helper_functions.write_to_md(text_mssg,None)
                        #load model
                        PATH = f'./model_scen_{0}_full.pth'
                        net = helper_functions.Load_Full_Model(PATH)

                        #external test image generation
                        train_loader, test_loader, stock_dataset_df = pipeline_data.generate_dataset_to_images_process(Parameters.external_test_stock_ticker, 
                                                                                    Parameters, 
                                                                                    Parameters.external_test_size, 
                                                                                    Parameters.external_test_cols_used)

                        #test
                        external_test_stack_input, external_test_stack_actual, external_test_stack_predicted = pipeline_test.test_process(net, 
                                                                                                                            test_loader, 
                                                                                                                            Parameters,
                                                                                                                            Parameters.external_test_stock_ticker)

                        #report stats
                        image_series_correlations, image_series_mean_correlation = external_test_pipeline.report_external_test_stats(
                                                                            Parameters, stock_dataset_df, 
                                                                            test_stack_input, external_test_stack_input,
                                                                            external_test_stack_actual, external_test_stack_predicted)

                        plot_data.plot_external_test_graphs(Parameters, test_stack_input, external_test_stack_input,
                                                    image_series_correlations, image_series_mean_correlation)
                        

if __name__ == "__main__":
    brute_force_function()