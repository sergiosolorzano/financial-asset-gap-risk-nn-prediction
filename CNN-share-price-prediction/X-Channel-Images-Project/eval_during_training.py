#!/usr/bin/env python

import os
import sys
import mlflow.azure
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from datetime import datetime
import torch
#import numpy as np
import itertools
import pandas as pd
import numpy as np
import math

from fastdtw import fastdtw
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
import torch.nn.functional as F

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))

from parameters import Parameters

import helper_functions_dir.neural_network_enhanced as neural_network
import helper_functions_dir.plot_data as plot_data
import helper_functions_dir.helper_functions as helper_functions
import helper_functions_dir.process_price_series as process_price_series
import helper_functions_dir.credentials as credentials
import helper_functions_dir.compute_stats as compute_stats

from parameters import StockParams
import pipeline_data as pipeline_data
import pipeline_train as pipeline_train
import pipeline_test as pipeline_test
import evaluation_test_pipeline as evaluation_test_pipeline

import mlflow
from torchinfo import summary
import shap

def create_train_eval_stocks_obj():
    stock_params = StockParams()

    # run concat
    #stock_params.add_train_stock('CFG', '2021-12-06', '2023-01-25')
    #stock_params.add_train_stock('ZION', '2021-12-06', '2023-01-25')
    #stock_params.add_train_stock('PWBK', '2021-12-06', '2023-01-25')
    #stock_params.add_train_stock('KEY', '2021-12-06', '2023-01-25')
    stock_params.add_train_stock('FITB', '2021-12-06', '2023-01-25')
    #stock_params.add_train_stock('SIVBQ', '2021-12-06', '2023-01-25')
    # stock_params.add_train_stock('SICP', '2021-12-06', '2023-01-25')
    # stock_params.add_train_stock('ALLY', '2021-12-06', '2023-01-25')
    # stock_params.add_train_stock('CMA', '2021-12-05', '2023-01-25')
    # stock_params.add_train_stock('WAL', '2021-12-05', '2023-01-25')
    
    #scenarios
    #stock_params.add_eval_stock('RF', '2021-12-06', '2023-01-25') 
    #stock_params.add_eval_stock('KEY', '2021-12-06', '2023-01-25') 
    #stock_params.add_eval_stock('OZK', '2021-12-06', '2023-01-25') 
    stock_params.add_eval_stock('CFG', '2021-12-06', '2023-01-25') 
    #stock_params.add_eval_stock('CUBI', '2021-12-06', '2023-01-25')
    #stock_params.add_eval_stock('WAL', '2021-12-06', '2023-01-25') 
    #stock_params.add_eval_stock('ZION', '2021-12-06', '2023-01-25')
    #stock_params.add_eval_stock('SICP', '2021-12-06', '2023-01-25')
    #stock_params.add_eval_stock('SIVBQ', '2021-12-06', '2023-01-25')
    #stock_params.add_eval_stock('ALLY', '2021-12-06', '2023-01-25')
    #stock_params.add_eval_stock('PWBK', '2021-12-06', '2023-01-25')

    stock_params.set_param_strings()
    Parameters.train_tickers = stock_params.train_stock_tickers
    Parameters.eval_tickers = stock_params.eval_stock_tickers

    return stock_params

class StockParams:
    def __init__(self):
        self.train_stocks = []
        self.eval_stocks = []
        self.train_stock_tickers = ""
        self.eval_stock_tickers = ""
        self.train_count = 0
        self.eval_count = 0

    def add_train_stock(self, ticker, start_date, end_date):
        stock_info = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        }
        self.train_stocks.append(stock_info)
        self.train_count +=1

    def add_eval_stock(self, ticker, start_date, end_date):
        stock_info = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        }
        self.eval_stocks.append(stock_info)
        self.eval_count +=1

    def get_train_stocks(self):
        return self.train_stocks
    
    def get_eval_stocks(self):
        return self.eval_stocks
    
    def set_param_strings(self):
        train_stocks = self.get_train_stocks()
        eval_stocks = self.get_eval_stocks()
        for s in train_stocks:
            self.train_stock_tickers = "_".join([s['ticker'] for s in train_stocks])
        for s in eval_stocks:
            self.eval_stock_tickers = "_".join([s['ticker'] for s in eval_stocks])
    
    def get_dates_by_ticker(self, ticker):
        for stock in self.train_stocks:
            if stock['ticker'] == ticker:
                return stock['start_date'], stock['end_date']
        for stock in self.eval_stocks:
            if stock['ticker'] == ticker:
                return stock['start_date'], stock['end_date']
        return None
    
    def get_all_tickers(self):
        all_stocks = self.train_stocks + self.eval_stocks  # Combine both lists
        return [stock['ticker'] for stock in all_stocks] 

def generate_shap(net, device, train_feature_maps_tensor, eval_feature_maps_tensor):
    
    print("AT generate_shap len",len(eval_feature_maps_fc_np),"shape",eval_feature_maps_fc_np.shape)

    feature_maps_shape = train_feature_maps_tensor.shape
    background = torch.randn(feature_maps_shape[0],feature_maps_shape[1],feature_maps_shape[2],feature_maps_shape[3])  # Background data (random noise or representative images)
    explainer = shap.DeepExplainer(net, background)

    train_feature_maps_tensor = train_feature_maps_tensor.to(device)
    eval_feature_maps_tensor = eval_feature_maps_tensor.to(device)

    # Generate SHAP values
    shap_values_train = explainer.shap_values(train_feature_maps_tensor)
    shap_values_eval = explainer.shap_values(eval_feature_maps_tensor)

    shap.image_plot([shap_values_train], [shap_values_train.cpu().numpy()])
    shap.image_plot([shap_values_eval], [shap_values_eval.cpu().numpy()])

def calculate_images_ssim(train_feature_image_dataset_list_f32_tensor, test_feature_image_dataset_list_f32_tensor, feature_type):
    data_range = max(torch.max(torch.abs(train_feature_image_dataset_list_f32_tensor)).item(),torch.max(torch.abs(test_feature_image_dataset_list_f32_tensor)).item())
    ssim_score = ssim(train_feature_image_dataset_list_f32_tensor, test_feature_image_dataset_list_f32_tensor, data_range=data_range)
    print(f"ssim_score {feature_type}: ",ssim_score)

    image_similarity = {"SSIM":ssim_score.item()}

def train(device) :
    
    feature_maps_cnn_list =[]
    feature_maps_fc_list =[]

    stock_params = create_train_eval_stocks_obj()
    
    train_loader, test_loader, evaluation_test_stock_dataset_df, train_feature_image_dataset_list_f32 = pipeline_data.generate_dataset_to_images_process(stock_params, stock_params.get_train_stocks(), 
                                                                                                                                                        Parameters, 
                                                                                                                                                        Parameters.training_test_size, 
                                                                                                                                                        Parameters.training_cols_used,
                                                                                                                                                        None, None)
    if Parameters.load_checkpoint_for_eval:
        net = neural_network.instantiate_net(Parameters, device)
        net, epoch, loss = helper_functions.load_checkpoint_model(net, device, stock_params)
        net  = neural_network.set_model_for_eval(net)
        torch.set_grad_enabled(False)

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            outputs, feature_map_cnn, feature_map_fc = net(inputs)
            #print("outputs",outputs)
            feature_maps_cnn_list.append(feature_map_cnn)
            feature_maps_fc_list.append(feature_map_fc)

    return feature_maps_cnn_list, feature_maps_fc_list

def eval(device):
    #################################
    #       Evaluation Test         #
    #################################

    feature_maps_cnn_list = []
    feature_maps_fc_list = []

    stock_params = create_train_eval_stocks_obj()
    
    #load best checkpoint
    if Parameters.load_checkpoint_for_eval:
        net  = neural_network.instantiate_net(Parameters, device)
        net, epoch, loss = helper_functions.load_checkpoint_model(net, device, stock_params)
        net  = neural_network.set_model_for_eval(net)
        torch.set_grad_enabled(False)

    #external test image generation
    train_loader, test_loader, evaluation_test_stock_dataset_df, test_feature_image_dataset_list_f32 = pipeline_data.generate_dataset_to_images_process(stock_params, stock_params.get_eval_stocks(), 
                                                                                                        Parameters,
                                                                                                        Parameters.evaluation_test_size, 
                                                                                                        Parameters.evaluation_test_cols_used,
                                                                                                        None, None)
    
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            outputs, feature_map_cnn, feature_map_fc = net(inputs)
            #print("outputs",outputs)
            feature_maps_cnn_list.append(feature_map_cnn)
            feature_maps_fc_list.append(feature_map_fc)

    #eval
    evaluation_test_stack_input, evaluation_test_stack_actual, evaluation_test_stack_predicted = pipeline_test.test_process(net, 
                                                                                                        test_loader, 
                                                                                                        Parameters,
                                                                                                        Parameters.eval_tickers, None,
                                                                                                        None, device)
    return feature_maps_cnn_list, feature_maps_fc_list, net

if __name__ == "__main__":
    
    device = torch.device("cpu")
    print(torch.__version__)
    
    train_feature_maps_cnn_list, train_feature_maps_fc_list = train(device)
    eval_feature_maps_cnn_list, eval_feature_maps_fc_list, net = eval(device)
    
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

    #generate_shap(net, device, eval_feature_maps_np, train_feature_maps_np)