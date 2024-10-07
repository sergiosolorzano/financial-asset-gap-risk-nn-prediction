from sklearn.preprocessing import StandardScaler,MinMaxScaler
import sys
import os
from enum import Enum
import pandas as pd
from datetime import datetime
import torch.nn as nn
import torch

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransformAlgo(Enum):
    GRAMIAN = 1
    MARKOV = 2

    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                #print("received ",value,"matching", member.value, "member", member)
                return member
        raise ValueError(f"Unsupported value: {value}")

class StockParams:
    def __init__(self):
        self.train_stocks = []
        self.eval_stocks = []
        self.train_stock_tickers = ""
        self.eval_stock_tickers = ""
        # self.start_date = ''
        # self.end_date = ''

    def add_train_stock(self, ticker, start_date, end_date):
        stock_info = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        }
        self.train_stocks.append(stock_info)

    def add_eval_stock(self, ticker, start_date, end_date):
        stock_info = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        }
        self.eval_stocks.append(stock_info)

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
    
#init parameters
class Parameters:
    matplotlib_use = "Agg"
    scenario = 0 #local txt logging param
    nn_predict_price = 1 #0=classification;1=regression
    classification_class_price_down=0
    classification_class_price_up=1

    log_returns = False #1=log return rebased price series else price series

    enable_mlflow=True
    mlflow_experiment_name = 'gaprisk-concatstocks-2'
    mlflow_experiment_description = "Concat stocks to train v2"
    
    brute_force_filename = 'brute_force_results.md'
    mlflow_credentials_fname = 'mlflow-creds.json'
    input_price_data_blob_fname = 'input_price_data_run_id'
    input_image_data_blob_fname = 'input_image_data_run_id'
    predicted_image_data_blob_fname = 'predicted_image_data_run_id'
    model_arch_fname = "model_arch"
    model_checkpoint_fname = "model_best_checkpoint"
    model_full_fname = 'full_model'

    save_runs_to_md = False

    save_arch_bool = True #only once

    plt_image_size = (12,5)
    
    mlflow_system_log_freq = 180 #secs

    brute_force_image_mlflow_dir = 'brute_force_images_mlflow'
    checkpoint_dir = 'models/model_checkpoints'
    full_model_dir = 'models/full_models'
    model_arch_dir = 'models/architecture_models'

    # Stock tickers
    train_tickers = ""
    eval_tickers = ""
    index_ticker = '^SP500-40'
    # start_date = ''
    # end_date = ''

    #cols used
    training_cols_used = ["Open", "High", "Low", "Close"]
    evaluation_test_cols_used = ["Open", "High"]

    # Time series to image transformation algorithm: GRAMIAN 1; MARKOV 2
    transform_algo_type = 1
    transform_algo = TransformAlgo.from_value(transform_algo_type)
    image_resolution_x = 32
    image_resolution_y = 32
    
    # GAF image inputs
    gaf_method = "summation"
    transformed_img_sz = 32
    gaf_sample_range = (-1, 0.5)
    
    # label scaler applied to labels only for both GRAMIAN/MARKOV
    scaler = StandardScaler()
    min_max_scaler_feature_range = (-1, 0) #for MinMaxScaler()

    # Training's test size
    training_test_size = 0.5 #50% to capture two full features
    evaluation_test_size = 1

    model_name ='LeNet-5 Based Net'
    # Default hyperparameters
    filter_size_1 = (2, 3)
    filter_size_2 = (2, 2)
    filter_size_3 = (2, 3)

    stride_1 = 1
    stride_2 = 2

    output_conv_1 = 40
    output_conv_2 = 12
    output_FC_1 = 100
    output_FC_2 = 70

    if nn_predict_price:
        #next day price
        final_FCLayer_outputs = 1
    else:
        #higher or lower price
        final_FCLayer_outputs = 2

    learning_rate = 0.001
    momentum = 0.9

    dropout_probab = 0

    batch_size = 16
    num_workers = 0

    num_epochs_input = 20000

    best_checkpoint_cum_loss = 0.002
    min_best_cum_loss = torch.tensor(2.5, device=device, dtype=torch.float64)

    loss_stop_threshold = 0.000001

    #adamw optimizer and cyclic scheduler
    run_adamw = False
    adamw_weight_decay = 0.00001
    adamw_scheduler_cyclic_policy = "cosine" #["cosine", "arccosine", "triangular", "triangular2", "exp_range"]
    adamw_scheduler_restart_period = 5 #epoch count in the first restart period
    adamw_scheduler_t_mult = 1.2 #multiplication factor by which the next restart period will expand/shrink
    
    #pytorch LR scheduler
    lr_scheduler_patience = 700 #10000 to ignore lrscheduler
    lr_scheduler_mode = 'min'
    #max_stale_loss_epochs = max(4 * lr_scheduler_patience,300)
    max_stale_loss_epochs = 300
    #max_stale_loss_epochs =100000 #to ignore lrscheduler
    # LR reset
    enable_lr_reset = True
    lr_reset_rate = 0.001
    lr_reset_threshold = 0.000001

    epoch_running_loss_check = 2500
    
    epoch_running_gradients_check = 4500

    checkpt_dict = {
            'run_id': None,
            'epoch': None,
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'loss': None,
            }
    
    if nn_predict_price:
        function_loss = nn.MSELoss()
    else:
        function_loss = nn.CrossEntropyLoss()

    optimizer = "SGD" #SGD or Adam if adamw=False, else adamw