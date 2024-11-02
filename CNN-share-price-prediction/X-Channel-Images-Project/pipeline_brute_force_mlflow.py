#!/usr/bin/env python

import os
import sys
import math
import mlflow.azure
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from datetime import datetime
import torch
#import numpy as np
import itertools
import pandas as pd
import numpy as np

from fastdtw import fastdtw
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn.functional as F

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import helper_functions_dir.neural_network_enhanced as neural_network
import helper_functions_dir.plot_data as plot_data
import helper_functions_dir.helper_functions as helper_functions
import helper_functions_dir.process_price_series as process_price_series
import helper_functions_dir.credentials as credentials
import helper_functions_dir.compute_stats as compute_stats

from parameters import Parameters
from parameters import StockParams
import pipeline_data as pipeline_data
import pipeline_train as pipeline_train
import pipeline_test as pipeline_test
import evaluation_test_pipeline as evaluation_test_pipeline

import mlflow
#context execution for mlflow with statement
import contextlib
from torchinfo import summary

ssim_list = {}
mse_list = {}

def create_comparison_stocks_obj():
    stock_params = StockParams()

    # stock_params.add_train_stock('SIVBQ', '2021-12-05', '2023-01-25')
    # stock_params.add_train_stock('ALLY', '2021-12-05', '2023-01-25')
    # stock_params.add_train_stock('WAL', '2021-12-05', '2023-01-25')
    # stock_params.add_train_stock('CUBI', '2021-12-05', '2023-01-25')
    
    stock_params.add_train_stock('SIVBQ', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('SICP', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('ALLY', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('CMA', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('WAL', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('PWBK', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('ZION', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('KEY', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('CUBI', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('OZK', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('CFG', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('RF', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('FITB', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('HBAN', '2021-12-06', '2023-01-25')
    
    # stock_params.add_train_stock('FRC', '2021-12-05', '2023-01-25')
    # stock_params.add_train_stock('SBNY', '2021-12-05', '2023-01-25')
    #stock_params.add_train_stock('JPM', '2021-12-05', '2023-01-25')

    stock_params.set_param_strings()
    # Parameters.train_tickers = stock_params.train_stock_tickers
    # Parameters.eval_tickers = stock_params.eval_stock_tickers

    return stock_params

def create_train_eval_stocks_obj():
    stock_params = StockParams()

    # run concat
    stock_params.add_train_stock('CFG', '2021-12-06', '2023-01-25')
    #stock_params.add_train_stock('ZION', '2021-12-06', '2023-01-25')
    #stock_params.add_train_stock('PWBK', '2021-12-06', '2023-01-25')
    #stock_params.add_train_stock('KEY', '2021-12-06', '2023-01-25')
    #stock_params.add_train_stock('FITB', '2021-12-06', '2023-01-25')
    #stock_params.add_train_stock('SIVBQ', '2021-12-06', '2023-01-25')
    # stock_params.add_train_stock('SICP', '2021-12-06', '2023-01-25')
    # stock_params.add_train_stock('ALLY', '2021-12-06', '2023-01-25')
    # stock_params.add_train_stock('CMA', '2021-12-05', '2023-01-25')
    # stock_params.add_train_stock('WAL', '2021-12-05', '2023-01-25')
    
    #scenarios
    stock_params.add_eval_stock('RF', '2021-12-06', '2023-01-25') 
    #stock_params.add_eval_stock('KEY', '2021-12-06', '2023-01-25') 
    #stock_params.add_eval_stock('OZK', '2021-12-06', '2023-01-25') 
    #stock_params.add_eval_stock('CFG', '2021-12-06', '2023-01-25') #loss ?, acc 32%, r^2 0.15
    #stock_params.add_eval_stock('CUBI', '2021-12-06', '2023-01-25') #loss ?, acc 32%, r^2 0.15
    #stock_params.add_eval_stock('WAL', '2021-12-06', '2023-01-25') #loss 0.10, acc 34%, r^2 -0.15
    #stock_params.add_eval_stock('SICP', '2021-12-06', '2023-01-25')
    #stock_params.add_eval_stock('SIVBQ', '2021-12-06', '2023-01-25')
    #stock_params.add_eval_stock('ALLY', '2021-12-06', '2023-01-25')
    #stock_params.add_eval_stock('PWBK', '2021-12-06', '2023-01-25')

    #stock_params.add_eval_stock('SICP', '2021-12-06', '2023-01-25')
    #high correl
    #stock_params.add_eval_stock('CMA', '2021-12-05', '2023-01-25')
    #medium correl
    #stock_params.add_eval_stock('JPM', '2021-12-05', '2023-01-25')
    #low correl
    #stock_params.add_eval_stock('RF', '2021-12-05', '2023-01-25')
    #nil correl
    #stock_params.add_eval_stock('CROX', '2021-12-05', '2023-01-25')

    stock_params.set_param_strings()
    Parameters.train_tickers = stock_params.train_stock_tickers
    Parameters.eval_tickers = stock_params.eval_stock_tickers

    return stock_params

def calculate_ssim_train_eval(train_feature_maps_cnn_list, train_feature_maps_fc_list, eval_feature_maps_cnn_list, eval_feature_maps_fc_list):
    #CNN
    train_feature_maps_cnn_np = torch.cat(train_feature_maps_cnn_list, dim=0)
    eval_feature_maps_cnn_np = torch.cat(eval_feature_maps_cnn_list, dim=0)
    # train_feature_maps_np = train_feature_maps_np.unsqueeze(1).unsqueeze(1)
    # eval_feature_maps_np = eval_feature_maps_np.unsqueeze(1).unsqueeze(1)
    calculate_images_ssim(train_feature_maps_cnn_np, eval_feature_maps_cnn_np, "CNN")
    
    #FC
    train_feature_maps_fc_np = torch.cat(train_feature_maps_fc_list, dim=0)
    eval_feature_maps_fc_np = torch.cat(eval_feature_maps_fc_list, dim=0)
    #print("AT MAIN len FC feature maps",len(eval_feature_maps_fc_np),"shape",eval_feature_maps_fc_np.shape)
    
    total_elements = train_feature_maps_fc_np.numel()
    width = int(math.sqrt(total_elements))
    height = total_elements // width

    while total_elements % width != 0:
        width -= 1
        height = total_elements // width
    train_feature_maps_fc_np = train_feature_maps_fc_np.view(width, height).unsqueeze(0).unsqueeze(0)
    eval_feature_maps_fc_np = eval_feature_maps_fc_np.view(width, height).unsqueeze(0).unsqueeze(0)
    calculate_images_ssim(train_feature_maps_fc_np, eval_feature_maps_fc_np, "FC")

def calculate_images_ssim(train_feature_image_dataset_list_f32, test_feature_image_dataset_list_f32, layer_type):
    
    print("train_feature_image_dataset_list_f32",train_feature_image_dataset_list_f32.dtype, train_feature_image_dataset_list_f32.shape)
    print("test_feature_image_dataset_list_f32",test_feature_image_dataset_list_f32.dtype, test_feature_image_dataset_list_f32.shape)
    if isinstance(train_feature_image_dataset_list_f32, np.ndarray):
        train_feature_image_dataset_list_f32_tensor = torch.from_numpy(train_feature_image_dataset_list_f32).unsqueeze(1)
    else:
        train_feature_image_dataset_list_f32_tensor = train_feature_image_dataset_list_f32

    if isinstance(test_feature_image_dataset_list_f32, np.ndarray):
        test_feature_image_dataset_list_f32_tensor = torch.from_numpy(test_feature_image_dataset_list_f32).unsqueeze(1)
    else:
        test_feature_image_dataset_list_f32_tensor = test_feature_image_dataset_list_f32
    #test_feature_image_dataset_list_f32_tensor = test_feature_image_dataset_list_f32_tensor
    #train_max = torch.max(torch.abs(train_feature_image_dataset_list_f32_tensor)).item()
    #eval_max = torch.max(torch.abs(test_feature_image_dataset_list_f32_tensor)).item()
    #data_range=max(train_max, eval_max)
    data_range = max(torch.max(torch.abs(train_feature_image_dataset_list_f32_tensor)).item(),torch.max(torch.abs(test_feature_image_dataset_list_f32_tensor)).item())
    ssim_score = ssim(train_feature_image_dataset_list_f32_tensor, test_feature_image_dataset_list_f32_tensor, data_range=data_range)
    
    if layer_type is not None:
        print(f"ssim_score {layer_type}: ",ssim_score)
        image_similarity = {f"SSIM_{layer_type}":ssim_score.item()}
    else:
        print(f"ssim_score Input Images: ",ssim_score)
        image_similarity = {"SSIM_Input_Images":ssim_score.item()}

    if Parameters.enable_mlflow:
        mlflow.log_metrics(image_similarity)

    return ssim_score.item()

# def dtw_images(train_images, test_images, train_tickers, eval_tickers):

#     results = []
    
#     distance, path = fastdtw(train_images, test_images)
#     #print("Comparing",stock_ticker,compare_ticker,"data",data_close[stock_ticker][:10],"VS",data_close[compare_ticker][:10])
#     results.append((train_tickers, eval_tickers, distance))
    
#     # Create DataFrame from the results
#     dtw_image_df = pd.DataFrame(results, columns=['Train_Ticker', 'Eval_Ticker', 'Distance'])
#     print("****", dtw_image_df)
#     return dtw_image_df

def calc_dtw_and_correl_logprices(stocks,run):
    
    plot_data.dtw_matrix_logprices(stocks, run, Parameters.mlflow_experiment_name)
    
    plot_data.plot_all_cross_correl_price_series(stocks, run, Parameters.mlflow_experiment_name)

def mlflow_log_params(curr_datetime, experiment_name, experiment_id, stock_params):
    
    #start_date_obj = datetime.strptime(stock_params.get_train_stocks()[0]['start_date'], '%Y-%m-%d')
    # start_date_obj = datetime.strptime(stock_params.start_date, '%Y-%m-%d')
    # end_date_obj = datetime.strptime(stock_params.end_date, '%Y-%m-%d')
    # print("DayCount~",(end_date_obj-start_date_obj))
    
    params_dict = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "iteration": curr_datetime,
        "train_stock_ticker": stock_params.train_stock_tickers,
        "eval_stock_ticker": stock_params.eval_stock_tickers,
        # "index_ticker": Parameters.index_ticker,
        # "start_date": stock_params.start_date,
        # "end_date": stock_params.end_date,
        # "daycount": end_date_obj-start_date_obj,
        "training_testing_cols_used": Parameters.training_cols_used,
        "training_test_size": Parameters.training_test_size,
        "evaluation_test_cols_used": Parameters.evaluation_test_cols_used,
        "evaluation_test_size": Parameters.evaluation_test_size,
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
        "output_conv_2": Parameters.output_conv_3,
        "output_conv_2": Parameters.output_conv_4,
        "output_FC_1": Parameters.output_FC_1,
        "output_FC_2": Parameters.output_FC_2,
        "final_FCLayer_outputs": Parameters.final_FCLayer_outputs,
        "learning_rate": Parameters.learning_rate,
        "momentum": Parameters.momentum,
        "dropout_probab": Parameters.dropout_probab,
        "batch_size": Parameters.batch_size,
        "num_workers": Parameters.num_workers,
        "num_epochs_input": Parameters.num_epochs_input,
        "max_stale_loss_epochs": Parameters.max_stale_loss_epochs,
        "loss_threshold": Parameters.loss_stop_threshold,
        "epoch_running_loss_check": Parameters.epoch_running_loss_check,
        "epoch_running_gradients_check": Parameters.epoch_running_gradients_check,
        "loss_function": Parameters.function_loss,
        "optimizer": Parameters.optimizer_type,
        "lr_scheduler_patience": Parameters.lr_scheduler_patience,
        "log_returns": Parameters.log_returns
        }

    mlflow.log_params(params_dict)

def set_mlflow_experiment(credentials):
    if Parameters.enable_mlflow:
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
    else:
        experiment_name = "Undefined"

    return experiment_name, experiment_id

def brute_force_function(credentials, device, stock_params):

    if Parameters.enable_mlflow:
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
    else:
        experiment_name = "Undefined"

    #transform_algo_types = [1,2]
    transform_algo_types = [1]
    #gaf_methods = ["summation", "difference"]
    gaf_methods = ["summation"]
    #scalers = [StandardScaler(), MinMaxScaler(feature_range=(-1, 1)), MinMaxScaler(feature_range=(0, 1))] #MinMaxScaler()
    #min_max_scaler_feature_range = [(-1, 0), (0, 1), (-1, 1)]
    min_max_scaler_feature_range = [(-1, 0)]
    scalers = [MinMaxScaler(min_max_scaler_feature_range[0])]
    #gaf_sample_ranges = [(-1, 0), (-1, 0.5), (-1, 1), (-0.5, 0), (-0.5, 0.5), (-0.5, 1), (0, 0.5), (0, 1)]
    #gaf_sample_ranges = [(-1, 0.5), (-1, 0), (-0.5, 0.5)]
    gaf_sample_ranges = [(-1, 0.5)]
    #dropout_probabs = [0.25, 0.5]
    dropout_probabs = [0]
    #gaf_sample_ranges = [(-1, 0.5)]
    batch_size_list=[16]#[16,32,64,128,256,512]
    num_workers = [0]#0,4,8,12,16
    img_size = [32]

    for i_s in img_size:
        for b in batch_size_list:
            for w in num_workers:
                for t in transform_algo_types:
                    for m in gaf_methods:
                        for d in dropout_probabs:
                            for sc in scalers:
                                for mm_sc in min_max_scaler_feature_range:
                                    for s in gaf_sample_ranges:

                                        curr_datetime = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                                        
                                        with (mlflow.start_run(run_name=f"run_{curr_datetime}") if Parameters.enable_mlflow else contextlib.nullcontext()) as run:
                                               
                                            #sys logs
                                            if Parameters.enable_mlflow:
                                                print(mlflow.MlflowClient().get_run(run.info.run_id).data)
                                            
                                                Parameters.run_id = run_id = run.info.run_id
                                                print("runid",run_id)
                                            else:
                                                run_id=None

                                            #plot all stocks
                                            stock_comp_params = create_comparison_stocks_obj()
                                            data_close, merged_df = process_price_series.log_rebase_dataset(stock_comp_params)
                                            plot_data.plot_merged_log_series(merged_df, experiment_name, run_id)

                                            #plot single training and eval stocks
                                            data_close, merged_df = process_price_series.log_rebase_dataset(stock_params)
                                            plot_data.plot_price_comparison_stocks(merged_df, experiment_name, run)
                                            
                                            #plot training (with concat) and eval stocks
                                            plot_data.plot_train_and_eval_df(stock_params,experiment_name,run)

                                            #calc and plot dwt and correl
                                            calc_dtw_and_correl_logprices(stock_comp_params,run)

                                            #calc this pair dtw logprices
                                            plot_data.calc_pair_dtw_distance(stock_params,experiment_name,run)
                                            
                                            Parameters.num_workers = w
                                            Parameters.batch_size = b
                                            Parameters.transform_algo_type = t
                                            Parameters.gaf_method = m
                                            Parameters.gaf_sample_range = s
                                            Parameters.scaler = sc
                                            Parameters.dropout_probab = d
                                            Parameters.min_max_scaler_feature_range = mm_sc
                                            Parameters.image_resolution_x = Parameters.image_resolution_y = Parameters.transformed_img_sz = i_s

                                            if Parameters.enable_mlflow:
                                                mlflow_log_params(curr_datetime, experiment_name, experiment_id, stock_params)
                                                if Parameters.save_runs_to_md:
                                                    helper_functions.write_to_md("<b><center>==========Optimization Iteration==========</center></b><p><p>",None)
                                                    #helper_functions.write_to_md(f"<p><b>transform_algo_type: {t} gaf_method: {m} gaf_sample_range: {s} scaler: {sc} dropout_probab: {d}</b><p>", None)
                                                    helper_functions.write_to_md(f"<p><b>gaf_sample_range: {s} scaler: {sc}</b><p>", None)
                                            
                                            #################################
                                            #       Train and Test          #
                                            #################################

                                            #generate training images
                                            train_loader, test_loader, evaluation_test_stock_dataset_df, train_feature_image_dataset_list_f32 = pipeline_data.generate_dataset_to_images_process(stock_params, stock_params.get_train_stocks(), 
                                                                                                                                                        Parameters, 
                                                                                                                                                        Parameters.training_test_size, 
                                                                                                                                                        Parameters.training_cols_used,
                                                                                                                                                        run, experiment_name)
                                        
                                            if Parameters.train:
                                                net, train_stack_input, train_feature_maps_cnn_list, train_feature_maps_fc_list = pipeline_train.train_process(train_loader, Parameters, run_id, experiment_name, device, stock_params)
                                                
                                                #test
                                                # set model to eval
                                                net  = neural_network.set_model_for_eval(net)

                                                if Parameters.training_test_size > 0:
                                                    test_stack_input, test_stack_actual, test_stack_predicted, test_feature_maps_cnn_list, test_feature_maps_fc_list = pipeline_test.test_process(net, test_loader, 
                                                                                                                        Parameters, 
                                                                                                                        Parameters.train_tickers, run,
                                                                                                                        experiment_name, device)

                                            #################################
                                            #       Evaluation Test         #
                                            #################################
                                            text_mssg= "<u><center>==========Run Evaluation Stock Tests:==========</center></u><p>"
                                            print("\n\n",text_mssg)
                                            if Parameters.save_runs_to_md:
                                                helper_functions.write_to_md(text_mssg,None)

                                            #load best checkpoint
                                            if Parameters.load_checkpoint_for_eval:
                                                net = neural_network.instantiate_net(Parameters, device)
                                                net, epoch, loss, checkpoint = helper_functions.load_checkpoint_model(net, device, stock_params)
                                                net  = neural_network.set_model_for_eval(net)
                                                torch.set_grad_enabled(False)
                                                #print("Parameters.checkpt_dict",Parameters.checkpt_dict['model_state_dict']['conv2.weight'])

                                                #load model
                                                #PATH = f'./model_scen_{0}_full.pth'
                                                #net = helper_functions.Load_Full_Model(PATH)

                                            #external test image generation
                                            print("NOW EVAL")
                                            train_loader, test_loader, evaluation_test_stock_dataset_df, test_feature_image_dataset_list_f32 = pipeline_data.generate_dataset_to_images_process(stock_params, stock_params.get_eval_stocks(), 
                                                                                                                                            Parameters, 
                                                                                                                                            Parameters.evaluation_test_size, 
                                                                                                                                            Parameters.evaluation_test_cols_used,
                                                                                                                                            run, experiment_name)
                                            for i, data in enumerate(test_loader, 0):
                                                inputs, labels = data[0].to(device), data[1].to(device)
                                            actual_tensor = labels.data
                                            print(f"Actual {i}",actual_tensor[:1])
                                            #dtw images DIFF
                                            # print("Train shape:", train_feature_image_dataset_list_f32.shape)
                                            # print("Test shape:", test_feature_image_dataset_list_f32.shape)
                                            # train_feature_image_dataset_list_f32=train_feature_image_dataset_list_f32.flatten() 
                                            # test_feature_image_dataset_list_f32 = test_feature_image_dataset_list_f32.flatten()
                                            # #train_feature_image_dataset_list_f32 = train_feature_image_dataset_list_f32[:len(test_feature_image_dataset_list_f32)]
                                            # print("Train size:", len(train_feature_image_dataset_list_f32))
                                            # print("Test size:", len(test_feature_image_dataset_list_f32))
                                            # diff = train_feature_image_dataset_list_f32 - test_feature_image_dataset_list_f32
                                            # print("DIFF:", diff)
                                            # # Optionally, compute the sum of differences
                                            # sum_diff = sum(diff)
                                            # print("Sum of differences:", sum_diff)
                                            
                                            #Calculate DTW for Images
                                            #dtw_image_df = dtw_images(train_feature_image_dataset_list_f32.flatten(), test_feature_image_dataset_list_f32.flatten(), stock_params.train_stock_tickers, stock_params.eval_stock_tickers)
                                            # print("TO CONCAT",dtw_image_df)
                                            # plot_data.dtw_matrix_encoded_images(dtw_image_df, stock_params, Parameters.run_id, Parameters.mlflow_experiment_name)
                                            # image_series_dtw_distance_df = pd.concat([image_series_dtw_distance_df, pd.DataFrame(dtw_image_df)], ignore_index=True)
                                            # print("AFTER CONCAT",image_series_dtw_distance_df)
                                            
                                            if Parameters.train and (len(train_feature_image_dataset_list_f32) == len(test_feature_image_dataset_list_f32)):
                                                #calculate structural similarity index measure for images
                                                ssim_list[f"Train:{stock_params.train_stock_tickers}_Eval:{stock_params.eval_stock_tickers}"]=(calculate_images_ssim(train_feature_image_dataset_list_f32, test_feature_image_dataset_list_f32, None))
                                                #print("train_feature_image_dataset_list_f32 [:2]",train_feature_image_dataset_list_f32[:2],"type",train_feature_image_dataset_list_f32.dtype,"shape",train_feature_image_dataset_list_f32.shape,"size",train_feature_image_dataset_list_f32.size)
                                                #print("type",train_feature_image_dataset_list_f32.dtype,"shape",train_feature_image_dataset_list_f32.shape,"size",train_feature_image_dataset_list_f32.size)
                                                #calculate MSE images
                                                mse=F.mse_loss(torch.from_numpy(train_feature_image_dataset_list_f32), torch.from_numpy(test_feature_image_dataset_list_f32)).item()
                                                mse_list[f"Train:{stock_params.train_stock_tickers}_Eval:{stock_params.eval_stock_tickers}"]=mse
                                                mse_dict = {"Images_MSE_LOSS":mse}
                                                if Parameters.enable_mlflow:
                                                    mlflow.log_metrics(mse_dict)
                                            
                                            #test
                                            evaluation_test_stack_input, evaluation_test_stack_actual, evaluation_test_stack_predicted, eval_feature_maps_cnn_list, eval_feature_maps_fc_list = pipeline_test.test_process(net, 
                                                                                                                                                test_loader, 
                                                                                                                                                Parameters,
                                                                                                                                                Parameters.eval_tickers, run,
                                                                                                                                                experiment_name, device)

                                            #report stats
                                            if Parameters.train:

                                                calculate_ssim_train_eval(train_feature_maps_cnn_list, train_feature_maps_fc_list, eval_feature_maps_cnn_list, eval_feature_maps_fc_list)

                                                image_series_correlations, image_series_mean_correlation = evaluation_test_pipeline.report_evaluation_test_stats(
                                                                                                    stock_params.get_eval_stocks(), Parameters, evaluation_test_stock_dataset_df, 
                                                                                                    train_stack_input, evaluation_test_stack_input,
                                                                                                    run, experiment_name)

                                                plot_data.plot_evaluation_test_graphs(Parameters, train_stack_input, evaluation_test_stack_input,
                                                                            image_series_correlations, image_series_mean_correlation,
                                                                            experiment_name, run_id)

                                                # #generate encoded image correlation matrix
                                                # if stock_params.train_count == stock_params.eval_count:
                                                #     new_data = {
                                                #         "Train_Stock": stock_params.train_stock_tickers.split('_'),
                                                #         "Eval_Stock": stock_params.eval_stock_tickers.split('_'),
                                                #         "Image_Correl": image_series_mean_correlation
                                                #     }
                                                #     new_df = pd.DataFrame(new_data)
                                                #     print("****CONCAT NOW")
                                                #     image_series_mean_correl_df = pd.concat([image_series_mean_correl_df, new_df], ignore_index=True)
                                                    
                                                #     if not Parameters.run_iter:
                                                #         plot_data.plot_encoded_image_correl_matrix(image_series_mean_correl_df, experiment_name, run_id)

                                                #     return image_series_mean_correl_df

def on_key_press(key):
    try:
        if key.char == 'e':  # Detect the 'e' key
            print("Key 'e' pressed, starting Evaluation Test...")
            evaluation_thread = threading.Thread(target=run_evaluation_test)
            evaluation_thread.start()
    except AttributeError:
        pass  # Non-character keys

if __name__ == "__main__":

    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['OMP_PROC_BIND'] = 'CLOSE'
    os.environ['OMP_SCHEDULE'] = 'dynamic'
    os.environ['GOMP_CPU_AFFINITY'] = '0-23'

    #sys logging
    # if Parameters.enable_mlflow:
    #     mlflow.enable_system_metrics_logging()
    #     mlflow.set_system_metrics_sampling_interval(Parameters.mlflow_system_log_freq)

    #set gpu env
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA Available:", torch.cuda.is_available())
    print("Num GPUs available",torch.cuda.device_count())
    print("At training - Device",device)
    print("cuda version",torch.version.cuda)
    print(torch.__version__)
    #track mem used by tensors
    torch.cuda.memory_allocated()

    _credentials = credentials.MLflow_Credentials()
    _credentials.get_credentials()

    #clean nn peer file
    with open(Parameters.training_analytics_params_log_fname, 'w') as file:
        file.write("")
    #iter
    if Parameters.run_iter:

        print("****RUNNING ITERATION****")
        tickers = [
        'SIVBQ', 'SICP', 'ALLY', 'CMA', 'WAL', 'PWBK', 'ZION', 'KEY', 
        'CUBI', 'OZK', 'CFG', 'RF', 'FITB', 'HBAN'
        ]
        # tickers = [
        # 'SIVBQ', 'SICP', 'KEY'
        # ]

        start_date = '2021-12-06'
        end_date = '2023-01-25'

        for train_stock, eval_stock in itertools.combinations(tickers, 2):
            
            stock_params = StockParams()
            stock_params.add_train_stock(train_stock, start_date, end_date)
            stock_params.add_eval_stock(eval_stock, start_date, end_date)
            
            stock_params.set_param_strings()
            Parameters.train_tickers = stock_params.train_stock_tickers
            Parameters.eval_tickers = stock_params.eval_stock_tickers
            
            brute_force_function(_credentials, device, stock_params)
    else:
        stock_params = create_train_eval_stocks_obj()
        brute_force_function(_credentials, device, stock_params)

    #if Parameters.run_iter:
        #print("ssim",ssim_list)
        #print("mse",mse_list)

        #hard coded dtws
        #dtw_combos.plot_dtw_pairs(Parameters.mlflow_experiment_name, Parameters.run_id)
        #plot_data.plot_encoded_image_correl_matrix(image_series_mean_correl_df, Parameters.mlflow_experiment_name, Parameters.run_id)
        #plot_data.plot_table_multiple_image_dtw_distance(image_series_dtw_distance_df, Parameters.mlflow_experiment_name, Parameters.run_id)

