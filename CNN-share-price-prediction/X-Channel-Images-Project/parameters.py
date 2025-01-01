from sklearn.preprocessing import StandardScaler,MinMaxScaler
import sys
import os
from enum import Enum
import pandas as pd
from datetime import datetime
import torch.nn as nn
import torch
import uuid

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
        self.train_count = 0
        self.eval_count = 0
        self.eval_start_date = None
        self.eval_end_date = None
        self.train_start_date = None
        self.train_end_date = None

    def add_train_stock(self, ticker, start_date, end_date):
        stock_info = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        }
        self.train_start_date = start_date
        self.train_end_date = end_date
        self.train_stocks.append(stock_info)
        self.train_count +=1

    def add_eval_stock(self, ticker, start_date, end_date):
        stock_info = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        }
        self.eval_start_date = start_date
        self.eval_end_date = end_date
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
    
#init parameters
class Parameters:
    matplotlib_use = "Agg"
    run_iter = False
    fine_tune = True
    freeze = True
    load_state_dict_strict_input = False
    load_state_dict_strict_used = load_state_dict_strict_input if fine_tune==True else True

    model_complexity = "Average" #Simple, Average, Complex
    train = True
    load_checkpoint_for_eval = True

    nn_predict_price = 1 #0=classification;1=regression
    classification_class_price_down=0
    classification_class_price_up=1

    log_returns = True #1=log return rebased price series else price series
    run_iter_multiple_sims = False

    enable_mlflow=False
    enable_save_model = False
    mlflow_experiment_name = 'gaprisk-LongTrain-Then-finetune'
    mlflow_experiment_description = "Train SICPQ-SIVBQ-2018 for less similar time series"
    run_id = None
    
    brute_force_filename = 'brute_force_results.md'
    mlflow_credentials_fname = 'mlflow-creds.json'
    input_price_data_blob_fname = 'input_price_data_run_id'
    input_image_data_blob_fname = 'input_image_data_run_id'
    predicted_image_data_blob_fname = 'predicted_image_data_run_id'
    model_arch_fname = "model_arch"
    model_arch_ft_fname = "model_arch_ft"
    model_checkpoint_fname = "model_best_checkpoint"
    model_full_fname = 'full_model'
    model_checkpoint_ft_fname = "model_best_checkpoint_ft"
    model_full_ft_fname = 'full_model_ft'

    save_runs_to_md = False
    extended_train_eval_reports = False #TODO

    save_arch_bool = True #only once

    plt_image_size = (12,5)
    
    mlflow_system_log_freq = 180 #secs

    brute_force_image_mlflow_dir = 'brute_force_images_mlflow'
    checkpoint_dir = 'models/model_checkpoints'
    full_model_dir = 'models/full_models'
    model_arch_dir = 'models/architecture_models'

    checkpoint_ft_dir = 'models/model_checkpoints_ft'
    full_model_ft_dir = 'models/full_models_ft'
    model_arch_ft_dir = 'models/architecture_models_ft'

    # ticker lists
    train_tickers = ""
    eval_tickers = ""

    #price data cols used
    training_cols_used = ["Close"] #, open, high, "Low", "Close"
    evaluation_test_cols_used = ["Close"] #, open, "High"

    # Time series to image transformation algorithm: GRAMIAN 1; MARKOV 2
    transform_algo_type = 1
    transform_algo = TransformAlgo.from_value(transform_algo_type)
    image_resolution_x = 32
    image_resolution_y = 32
    
    # GAF image inputs
    gaf_method = "summation"
    transformed_img_sz = 32
    gaf_sample_range = (-1, 0.5)
    window_method = 2 # 1=overlapping , 2=myoverlap, 3 use full windowing
    window_overlap = 25 #overlapping datapoints at the beginning of window between windows 
    
    # label scaler applied to labels only for both GRAMIAN/MARKOV
    min_max_scaler_feature_range = (-1, 0) #for MinMaxScaler()
    scaler = MinMaxScaler(min_max_scaler_feature_range)#StandardScaler()

    # Training's test size
    training_test_size = 0 #size of the test dataset when training: 50% of first feature/s, 50% of the next
    evaluation_test_size = 1 #100% of the feature

    model_name ='LeNet-5 Based Net'
    # Default hyperparameters
    # filter_size_1 = (3, 3)#(2, 3)
    # filter_size_2 = (2, 2)#(2,2)
    # filter_size_3 = (1,1)#(2, 3)

    stride_1 = 1
    stride_2 = 1#2

    #TODO dynamic calc groups
    use_batch_regularization_conv = True
    use_batch_regularization_fc = True
    
    batch_regul_type_conv = "Norm2" #Norm2, Group
    batch_regul_type_fc = "LayerNorm" #Norm, Group, LayerNorm
    bn1_num_groups = 5  # Conv1: 40 channels / 5 groups = 8 channels per group
    bn2_num_groups = 3  # Conv2: 12 channels / 3 groups = 4 channels per group
    bn3_num_groups = 0  # Not used in "Average" complexity
    bn4_num_groups = 0  # Not used in "Average" complexity
    bn_fc1_num_groups = 10  # FC1: 100 channels / 10 groups = 10 channels per group
    bn_fc2_num_groups = 10  # FC2: 70 channels / 10 groups = 7 channels per group

    use_adaptiveAvgPool2d = False
    adaptiveAvgPool2d_outputsize = (1,1)

    if model_complexity=="Complex":
        filter_size_1 = (3, 3)
        filter_size_2 = (2, 2)
        filter_size_3 = (1,1)

        output_conv_1 = 64
        output_conv_2 = 128
        output_conv_3 = 256
        output_conv_4 = 512
        output_FC_1 = 100
        output_FC_2 = 100

    if model_complexity == "Average":
        filter_size_1 = (2, 3)#(2, 3)
        filter_size_2 = (2,2)
        filter_size_3 = (2, 3)

        output_conv_1 = 40
        output_conv_2 = 12
        output_conv_3 = 0
        output_conv_4 = 0
        output_FC_1 = 100
        output_FC_2 = 70

    if model_complexity == "Simple":
        filter_size_1 = (1, 1)#(2, 3)
        filter_size_2 = (2,2)
        filter_size_3 = (2, 3)

        output_conv_1 = 40
        output_conv_2 = 30
        output_conv_3 = 0
        output_conv_4 = 0
        output_FC_1 = 100
        output_FC_2 = 0
        
    if nn_predict_price:
        #next day price
        final_FCLayer_outputs = 1
    else:
        #higher or lower price
        final_FCLayer_outputs = 2

    learning_rate = 0.001
    #Optimizer Layers LR
    use_layer_lr = True
    conv_lr = 0.001
    fc_lr = 0.01

    momentum_sgd = 0.9

    ssim_list = []

    use_ssim_adjusted_loss = False
    lambda_ssim = 0.5
    cnn_fc_lambda_ssim_ratio = 0.2

    use_clip_grad_norm = True
    grad_norm_clip_max = 5

    dropout_probab_1 = 0
    dropout_probab_2 = 0

    batch_size = 16
    batch_train_drop_last = False
    batch_eval_drop_last = False
    num_workers = 0

    best_avg_checkpoint_cum_loss = 5
    min_best_cum_loss = torch.tensor(2.5, device=device, dtype=torch.float64)
    model_uuid = 1#str(uuid.uuid4())

    loss_stop_threshold = 0.000001#0.000001#0.0000013
    use_mixed_precision = False

    #weights init
    kaiming_uniform_nonlinearity_type = 'relu' #relu leaky_relu
    kaiming_uniform_leakyrelu_a = 0

    #pytorch Schedulers
    scheduler_type = "CyclicLRWithRestarts" #ReduceLROnPlateau, OneCycleLR, CyclicLRWithRestarts, BayesianLR, Warmup
    scheduler = None    

    #Warmup Scheduler
    start_factor=0.001
    total_iters = 100
    
    #bayesian Scheduler
    bayesianLR_bounds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-6, 1e-1)}]
    bayesian_warmup_learning_rates = [0.0001, 0.00001, 0.000001, 0.001, 0.01]
    bayesian_warmup_epochs = 5
    bayes_find_lr_frequency_epochs = 45+bayesian_warmup_epochs
    bayes_loss_threshold_to_log = 0.6

    #CyclicLRWithRestarts Scheduler
    cyclicLRWithRestarts_cyclic_policy = "cosine" #["cosine", "arccosine", "triangular", "triangular2", "exp_range"]
    cyclicLRWithRestarts_restart_period = 15 #epoch count in the first restart period
    cyclicLRWithRestarts_t_mult = 1.2 #multiplication factor by which the next restart period will expand/shrink
    cyclicLRWithRestarts_min_lr = 0.000001 #default 0.0000001

    #ReduceLROnPlateau Scheduler:
    reduceLROnPlateau_patience = 15 #100 #10000 to ignore lrscheduler
    reduceLROnPlateau_reset_cooldown = 7 #30
    reduceLROnPlateau_mode = 'min'
    reduceLROnPlateau_enable_reset = True
    reduceLROnPlateau_reset_rate = 0.1#manual reset rate, will be ratched down by factor in scheduler.step
    reduceLROnPlateau_factor = 0.1
    reduceLROnPlateau_min_lr = 1e-3 #can only be higher than 1e-8 

    #OneCycleLR:
    oneCycleLR_max_lr=0.01
    oneCycleLR_pct_start=0.12 #percentage of the cycle spent increasing the learning rate

    #select optimizer
    optimizer_type = "adam.Adamw" #SGD , Adam, adam.Adamw, optim.Adamw
    optimizer = None

    #TODO refactor configs optimizers and schedulers
    #adamw optimizer
    adamw_weight_decay = 0.0001#0.00001#0.000001
    
    #adam optimizer
    adam_weight_decay = 0.00001
    adam_betas = (0.9,0.999) #(exponential decay rate for the first moment estimate,exponential decay rate for the first moment estimate). Either higher more responsive.

    epoch_running_loss_check = 2500
    
    epoch_running_gradients_check = 12000

    checkpt_dict = {
            'run_id': None,
            'epoch': None,
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'loss': None,
            }
    
    if nn_predict_price:
        #function_loss = nn.MSELoss()
        function_loss = nn.L1Loss()
        #function_loss = nn.SmoothL1Loss()
    else:
        function_loss = nn.CrossEntropyLoss()

    #regularization activation funcs
    use_relu = False
    if use_relu:
        regularization_function = nn.ReLU()
    else:
        regularization_function = nn.SiLU()

    #during training-and-eval vars
    num_epochs_input = 1000
    eval_at_epoch_multiple = 1
    save_checkpoint = True
    save_checkpoint_at_epoch_multiple = 10
    log_params_at_epoch_multiple = 1
    log_weights = True
    training_analytics_params_log_fname = 'nn_peer_stats.txt'

    #global tracking vars
    max_acc_1dp = torch.tensor(0, dtype=torch.float64)
    max_acc_1dp_epoch = 0
    best_eval_r2 = torch.tensor(0, dtype=torch.float64)
    best_eval_r2_epoch = 0
    train_max_r2 = 0
    train_max_r2_epoch = 0
    eval_max_r2 = 0
    eval_max_r2_epoch = 0

    fc_ssim_score = torch.tensor(0, dtype=torch.float64)
    cnn_ssim_score = torch.tensor(0, dtype=torch.float64)
