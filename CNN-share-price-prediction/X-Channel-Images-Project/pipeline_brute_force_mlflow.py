#!/usr/bin/env python

import os
import sys
import mlflow.azure
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from datetime import datetime

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import helper_functions_dir.neural_network as neural_network
import helper_functions_dir.plot_data as plot_data
import helper_functions_dir.helper_functions as helper_functions
import helper_functions_dir.compute_stats as compute_stats 
import helper_functions_dir.credentials as credentials

from parameters import Parameters
import pipeline_data as pipeline_data
import pipeline_train as pipeline_train
import pipeline_test as pipeline_test
import external_test_pipeline as external_test_pipeline

import mlflow
from torchinfo import summary

def mlflow_log_params(curr_datetime, experiment_name, experiment_id):
    
    start_date_obj = datetime.strptime(Parameters.start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(Parameters.end_date, '%Y-%m-%d')
    print("DayCount~",end_date_obj-start_date_obj)

    params_dict = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "iteration": curr_datetime,
        "train_stock_ticker": Parameters.train_stock_ticker,
        "external_test_stock_ticker": Parameters.external_test_stock_ticker,
        "index_ticker": Parameters.index_ticker,
        "start_date": Parameters.start_date,
        "end_date": Parameters.end_date,
        "daycount": end_date_obj-start_date_obj,
        "training_testing_cols_used": Parameters.training_cols_used,
        "training_test_size": Parameters.training_test_size,
        "external_test_cols_used": Parameters.external_test_cols_used,
        "external_test_size": Parameters.external_test_size,
        "transform_algo_type": Parameters.transform_algo_type,
        "transform_algo": Parameters.transform_algo,
        "scaler": str(type(Parameters.scaler).__name__),
        "min_max_scaler_feature_range": Parameters.min_max_scaler_feature_range,
        "image_resolution_x": Parameters.image_resolution_x,
        "image_resolution_y": Parameters.image_resolution_y,
        "gaf_method": Parameters.gaf_method,
        "gaf_sample_range": Parameters.gaf_sample_range,
        "transformed_img_sz": Parameters.transformed_img_sz,
        "model_name": Parameters.model_name,
        "filter_size_1": Parameters.filter_size_1,
        "filter_size_2": Parameters.filter_size_2,
        "filter_size_3": Parameters.filter_size_3,
        "stride_1": Parameters.stride_1,
        "stride_2": Parameters.stride_2,
        "output_conv_1": Parameters.output_conv_1,
        "output_conv_2": Parameters.output_conv_2,
        "output_FC_1": Parameters.output_FC_1,
        "output_FC_2": Parameters.output_FC_2,
        "final_FCLayer_outputs": Parameters.final_FCLayer_outputs,
        "learning_rate": Parameters.learning_rate,
        "momentum": Parameters.momentum,
        "dropout_probab": Parameters.dropout_probab,
        "batch_size": Parameters.batch_size,
        "num_epochs_input": Parameters.num_epochs_input,
        "max_stale_loss_epochs": Parameters.max_stale_loss_epochs,
        "loss_threshold": Parameters.loss_threshold,
        "epoch_running_loss_check": Parameters.epoch_running_loss_check,
        "epoch_running_gradients_check": Parameters.epoch_running_gradients_check,
        "loss_function": Parameters.function_loss,
        "optimizer": Parameters.optimizer
    }

    mlflow.log_params(params_dict)

def brute_force_function(credentials):

    #logging.basicConfig(level=logging.DEBUG)
    #os.environ["AZURE_STORAGE_CONNECTION_STRING"] = f"set AZURE_STORAGE_CONNECTION_STRING={credentials.AZURE_STORAGE_CONNECTION_STRING}"
    os.environ["AZURE_STORAGE_ACCESS_KEY"] = credentials.AZURE_STORAGE_ACCESS_KEY
    mlflow.set_tracking_uri(f"mssql+pymssql://{credentials.username_db}:{credentials.password_db}@{credentials.server_db}.database.windows.net/{credentials.mlflow_db}")

    #create experiment
    experiment_name = Parameters.mlflow_experiment_name
    experiment_description = (Parameters.mlflow_experiment_description)
    experiment_tags = {"mlflow.note.content": experiment_description,}
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
        print(f"Experiment {experiment} id {experiment_id} exists.")
    else:
        mlflow.create_experiment(Parameters.mlflow_experiment_name, artifact_location=
                                 f"wasbs://{credentials.storage_container_name}@{credentials.storage_account_name}.blob.core.windows.net/{credentials.blob_directory}",#wasbs://<container>@<storage_account_name>.blob.core.windows.net/<dir>
                                 tags=experiment_tags)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Create experiment {experiment} id {experiment_id}")
    
    mlflow.set_experiment(experiment_name)
    
    #transform_algo_types = [1,2]
    transform_algo_types = [1]
    #gaf_methods = ["summation", "difference"]
    gaf_methods = ["summation"]
    #scalers = [StandardScaler(), MinMaxScaler(feature_range=(-1, 1)), MinMaxScaler(feature_range=(0, 1))] #MinMaxScaler()
    scalers = [MinMaxScaler(Parameters.min_max_scaler_feature_range)]
    min_max_scaler_feature_range = [(-1, 0), (0, 1), (-1, 1)]
    #min_max_scaler_feature_range = [(-1, 0)]
    #gaf_sample_ranges = [(-1, 0), (-1, 0.5), (-1, 1), (-0.5, 0), (-0.5, 0.5), (-0.5, 1), (0, 0.5), (0, 1)]
    gaf_sample_ranges = [(-1, 0.5), (-1, 0), (-0.5, 0.5)]
    dropout_probabs = [0.25, 0.5]

    for t in transform_algo_types:
        for m in gaf_methods:
            for d in dropout_probabs:
                for sc in scalers:
                    for mm_sc in min_max_scaler_feature_range:
                        for s in gaf_sample_ranges:

                            curr_datetime = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

                            with mlflow.start_run(run_name=f"run_{curr_datetime}") as run:

                                run_id = run.info.run_id
                                print("runid",run_id)

                                Parameters.transform_algo_type = t
                                Parameters.gaf_method = m
                                Parameters.gaf_sample_range = s
                                Parameters.scaler = sc
                                Parameters.dropout_probab = d
                                Parameters.min_max_scaler_feature_range = mm_sc

                                mlflow_log_params(curr_datetime, experiment_name, experiment_id)

                                if Parameters.save_runs_to_md:
                                    helper_functions.write_to_md("<b><center>==========Optimization Iteration==========</center></b><p><p>",None)
                                    #helper_functions.write_to_md(f"<p><b>transform_algo_type: {t} gaf_method: {m} gaf_sample_range: {s} scaler: {sc} dropout_probab: {d}</b><p>", None)
                                    helper_functions.write_to_md(f"<p><b>gaf_sample_range: {s} scaler: {sc}</b><p>", None)
                                
                                #################################
                                #       Train and Test          #
                                #################################

                                #generate training images
                                train_loader, test_loader, external_test_stock_dataset_df = pipeline_data.generate_dataset_to_images_process(Parameters.train_stock_ticker, 
                                                                                            Parameters, 
                                                                                            Parameters.training_test_size, 
                                                                                            Parameters.training_cols_used,
                                                                                            run, experiment_name)

                                net, train_stack_input = pipeline_train.train_process(train_loader, Parameters, run_id, experiment_name)

                                #test
                                # set model to eval
                                net  = neural_network.set_model_for_eval(net)

                                test_stack_input, test_stack_actual, test_stack_predicted = pipeline_test.test_process(net, test_loader, 
                                                                                                        Parameters, 
                                                                                                        Parameters.train_stock_ticker, run,
                                                                                                        experiment_name)

                                #################################
                                #       External Test           #
                                #################################
                                text_mssg= "<u><center>==========Run External Stock Tests:==========</center></u><p>"
                                print("\n\n",text_mssg)
                                if Parameters.save_runs_to_md:
                                    helper_functions.write_to_md(text_mssg,None)
                                #load model
                                #PATH = f'./model_scen_{0}_full.pth'
                                #net = helper_functions.Load_Full_Model(PATH)

                                #external test image generation
                                train_loader, test_loader, external_test_stock_dataset_df = pipeline_data.generate_dataset_to_images_process(Parameters.external_test_stock_ticker, 
                                                                                            Parameters, 
                                                                                            Parameters.external_test_size, 
                                                                                            Parameters.external_test_cols_used,
                                                                                            run, experiment_name)

                                #test
                                external_test_stack_input, external_test_stack_actual, external_test_stack_predicted = pipeline_test.test_process(net, 
                                                                                                                                    test_loader, 
                                                                                                                                    Parameters,
                                                                                                                                    Parameters.external_test_stock_ticker, run,
                                                                                                                                    experiment_name)

                                #report stats
                                image_series_correlations, image_series_mean_correlation = external_test_pipeline.report_external_test_stats(
                                                                                    Parameters, external_test_stock_dataset_df, 
                                                                                    train_stack_input, external_test_stack_input,
                                                                                    run, experiment_name)

                                plot_data.plot_external_test_graphs(Parameters, train_stack_input, external_test_stack_input,
                                                            image_series_correlations, image_series_mean_correlation,
                                                            experiment_name, run_id)
                                
if __name__ == "__main__":
    _credentials = credentials.MLflow_Credentials()
    _credentials.get_credentials()
    brute_force_function(_credentials)