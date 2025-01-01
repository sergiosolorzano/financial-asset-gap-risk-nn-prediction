from __future__ import print_function

import os
import sys
import time
import gc
import re
import GPUtil
import random

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn
from torchmetrics import R2Score, Accuracy,MeanAbsoluteError
import GPyOpt

import numpy as np
from sklearn.metrics import confusion_matrix
from torchmetrics.functional import structural_similarity_index_measure as ssim

import mlflow

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import adamw as adamw
import cyclic_scheduler as cyclic_scheduler 
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import LinearLR
import plot_data as plot_data
import image_transform as image_transform
import helper_functions as helper_functions
import compute_stats as compute_stats
import pipeline_brute_force_mlflow as pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parameters import Parameters

from torch.profiler import profile, ProfilerActivity
import contextlib

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)

class Net(nn.Module):
    def __init__(self, params, device):
        super(Net, self).__init__()
        
        if params.model_name:
            self.name = Parameters.model_name
        self.totalparams = 0
        self.output_conv_1= Parameters.output_conv_1
        self.output_conv_2= Parameters.output_conv_2
        self.output_conv_3= Parameters.output_conv_3
        self.output_conv_4= Parameters.output_conv_4
        self.conv_output_size=0
        self.dropout_probab_1 = Parameters.dropout_probab_1
        self.dropout_probab_2 = Parameters.dropout_probab_2
        #print();print("Convos & dropoutP:", params.output_conv_1, params.output_conv_2, params.dropout_probab)
        
        print("###params.dropout_probab_1",Parameters.dropout_probab_1,"dropout_2",Parameters.dropout_probab_1)

        #num channels input, num channels output, filter size
        self.conv1 = nn.Conv2d(1, self.output_conv_1, Parameters.filter_size_1, Parameters.stride_1)
        if Parameters.use_batch_regularization_conv:
            if Parameters.batch_regul_type_conv=="Group":
                self.bn1 = nn.GroupNorm(Parameters.bn1_num_groups,self.output_conv_1)
            if Parameters.batch_regul_type_conv=="Norm2":
                self.bn1 = nn.BatchNorm2d(self.output_conv_1)

        self.regularization_activation_function_1 = Parameters.regularization_function
        self.pool1 = nn.MaxPool2d(kernel_size=Parameters.filter_size_2, stride = Parameters.stride_2)
        
        #maxpool acts the same way in each channel, so doesn't need to be fed the num channels of the input
        self.conv2 = nn.Conv2d(self.output_conv_1, Parameters.output_conv_2, Parameters.filter_size_1,Parameters.stride_1)
        
        if Parameters.use_batch_regularization_conv:
            if Parameters.batch_regul_type_conv=="Group":
                self.bn2 = nn.GroupNorm(Parameters.bn2_num_groups,self.output_conv_2)
            if Parameters.batch_regul_type_conv=="Norm2":
                self.bn2 = nn.BatchNorm2d(Parameters.output_conv_2)

        self.regularization_activation_function_2 = Parameters.regularization_function
        self.pool2 = nn.MaxPool2d(kernel_size=Parameters.filter_size_2, stride = Parameters.stride_2)

        if Parameters.use_adaptiveAvgPool2d:
            self.adaptive_pool = nn.AdaptiveAvgPool2d(Parameters.adaptiveAvgPool2d_outputsize)

        if Parameters.model_complexity == "Complex":
            self.conv3 = nn.Conv2d(Parameters.output_conv_2, Parameters.output_conv_3, Parameters.filter_size_1,Parameters.stride_1)
            if Parameters.use_batch_regularization_conv:
                if Parameters.batch_regul_type_conv == "Group":
                    self.bn3 = nn.GroupNorm(Parameters.bn3_num_groups,Parameters.output_conv_3)
                if Parameters.batch_regul_type_conv=="Norm2":
                    self.bn3 = nn.BatchNorm2d(Parameters.output_conv_3)

            self.regularization_activation_function_3 = Parameters.regularization_function
            self.pool3 = nn.MaxPool2d(kernel_size=Parameters.filter_size_2, stride = Parameters.stride_2)

            self.conv4 = nn.Conv2d(Parameters.output_conv_3, Parameters.output_conv_4, Parameters.filter_size_3,params.stride_1)

            if Parameters.use_batch_regularization_conv:
                if Parameters.batch_regul_type_conv == "Group":
                    self.bn4 = nn.GroupNorm(Parameters.bn4_num_groups,Parameters.output_conv_4)
                if Parameters.batch_regul_type_conv=="Norm2":
                    self.bn4 = nn.BatchNorm2d(Parameters.output_conv_4)
            
            self.regularization_activation_function_4 = Parameters.regularization_function
            self.pool4 = nn.MaxPool2d(kernel_size=Parameters.filter_size_2, stride = Parameters.stride_2)

        # After Conv1
        H_out_1, W_out_1 = image_transform.conv_output_shape_dynamic(
            (params.image_resolution_y, params.image_resolution_x),
            kernel_size=params.filter_size_1,
            stride=params.stride_1
        )

        # After Pool1
        H_out_2, W_out_2 = image_transform.conv_output_shape_dynamic(
            (H_out_1, W_out_1),
            kernel_size=params.filter_size_2,
            stride=params.stride_2
        )

        # After Conv2
        H_out_3, W_out_3 = image_transform.conv_output_shape_dynamic(
            (H_out_2, W_out_2),
            kernel_size=params.filter_size_1,
            stride=params.stride_1
        )  

        # After Pool2
        H_out_4, W_out_4 = image_transform.conv_output_shape_dynamic(
            (H_out_3, W_out_3),
            kernel_size=params.filter_size_2,
            stride=params.stride_2
        )

        if Parameters.model_complexity=="Complex":
            # After Conv3
            H_out_5, W_out_5 = image_transform.conv_output_shape_dynamic(
                (H_out_4, W_out_4),
                kernel_size=params.filter_size_1,
                stride=params.stride_1
            )  

            # After Pool3
            H_out_6, W_out_6 = image_transform.conv_output_shape_dynamic(
                (H_out_5, W_out_5),
                kernel_size=params.filter_size_2,
                stride=params.stride_2
            )

            # After Conv4
            H_out_7, W_out_7 = image_transform.conv_output_shape_dynamic(
                (H_out_6, W_out_6),
                kernel_size=params.filter_size_3,
                stride=params.stride_1 #use stride_1 as per conv3
            )  

            # After Pool4
            H_out_8, W_out_8 = image_transform.conv_output_shape_dynamic(
                (H_out_7, W_out_7),
                kernel_size=params.filter_size_2,
                stride=params.stride_2
            )

        # Update conv_output_size
        if Parameters.use_adaptiveAvgPool2d:
            # Set conv_output_size based on adaptive pooling output size
            self.conv_output_size = Parameters.adaptiveAvgPool2d_outputsize[0] * Parameters.adaptiveAvgPool2d_outputsize[1]
        else:
            if Parameters.model_complexity == "Complex":
                self.conv_output_size = H_out_8 * W_out_8
            else:
                self.conv_output_size = H_out_4 * W_out_4

        #FC Layers  
        if Parameters.use_adaptiveAvgPool2d:
            if Parameters.model_complexity == "Complex":
                fc1_in_features = Parameters.output_conv_4 * Parameters.adaptiveAvgPool2d_outputsize[0] * Parameters.adaptiveAvgPool2d_outputsize[1]
            else:
                fc1_in_features = Parameters.output_conv_2 * Parameters.adaptiveAvgPool2d_outputsize[0] * Parameters.adaptiveAvgPool2d_outputsize[1]
        else:
            if Parameters.model_complexity == "Complex":
                fc1_in_features = Parameters.output_conv_4 * self.conv_output_size
            else:
                fc1_in_features = Parameters.output_conv_2 * self.conv_output_size

        self.fc1 = nn.Linear(fc1_in_features, Parameters.output_FC_1)
        
        if Parameters.use_batch_regularization_fc:
                if Parameters.batch_regul_type_fc == "Group":
                    self.bn_fc1 = nn.GroupNorm(Parameters.bn_fc1_num_groups,Parameters.output_FC_1)
                if Parameters.batch_regul_type_fc == "Norm2":
                    self.bn_fc1 = nn.BatchNorm1d(Parameters.output_FC_1)
                if Parameters.batch_regul_type_fc=="LayerNorm":
                    self.bn_fc1 = nn.LayerNorm(Parameters.output_FC_1)

        self.regularization_activation_function_fc1 = Parameters.regularization_function
        
        if Parameters.model_complexity == "Average" or Parameters.model_complexity=="Complex":
            
            self.fc2 = nn.Linear(Parameters.output_FC_1, Parameters.output_FC_2)
            
            if Parameters.use_batch_regularization_fc:
                if Parameters.batch_regul_type_fc == "Group":
                    self.bn_fc2 = nn.GroupNorm(Parameters.bn_fc2_num_groups, Parameters.output_FC_2)
                if Parameters.batch_regul_type_fc == "Norm2":
                    self.bn_fc2 = nn.BatchNorm1d(Parameters.output_FC_2)
                if Parameters.batch_regul_type_fc =="LayerNorm":
                    self.bn_fc2 = nn.LayerNorm(Parameters.output_FC_2)
            
            self.regularization_activation_function_fc2 = Parameters.regularization_function
    
            self.fc3 = nn.Linear(Parameters.output_FC_2, Parameters.final_FCLayer_outputs)
        if Parameters.model_complexity=="Simple":
            self.fc3 = nn.Linear(Parameters.output_FC_1, Parameters.final_FCLayer_outputs)
        
        self.dropout1 = nn.Dropout(Parameters.dropout_probab_1)
        self.dropout2 = nn.Dropout(Parameters.dropout_probab_2)

        # compute the total number of parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #print(self.name + ': total params:', total_params)
        self.totalparams=total_params

    def forward(self, x):
        #Convo Layers
        if torch.isnan(self.conv1.weight).any() or torch.isinf(self.conv1.weight).any():
            print("NaN or Inf detected in conv1 weights before convolution")
            return x, None, None
        x = self.conv1(x)
        if Parameters.use_batch_regularization_conv:
            x = self.bn1(x)
        x = self.regularization_activation_function_1(x)
        if self.dropout_probab_1 > 0:
            x = self.dropout1(x)  
        x = self.pool1(x)
        
        x = self.conv2(x)
        if Parameters.use_batch_regularization_conv:
            x = self.bn2(x)
        x = self.regularization_activation_function_2(x)
        if self.dropout_probab_1 > 0:
            x = self.dropout1(x)  
        x = self.pool2(x)

        if Parameters.model_complexity == "Complex":
            x = self.conv3(x)
            if Parameters.use_batch_regularization_conv:
                x = self.bn3(x)
            x = self.regularization_activation_function_3(x)
            x = self.pool3(x)
            
            x = self.conv4(x)
            if Parameters.use_batch_regularization_conv:
                x = self.bn4(x)
            x = self.regularization_activation_function_4(x)
            x = self.pool4(x)

            #capture feature maps
            feature_maps_cnn = x

            if Parameters.use_adaptiveAvgPool2d:
                x = self.adaptive_pool(x)
                #Flatten for FC
                x = x.view(x.size(0), -1)
            else:
                x = x.view(-1, self.output_conv_4 * self.conv_output_size)
        
        if Parameters.model_complexity == "Average" or Parameters.model_complexity == "Simple":
            #capture feature maps
            feature_maps_cnn = x

            if Parameters.use_adaptiveAvgPool2d:
                x = self.adaptive_pool(x)
                #Flatten for FC
                x = x.view(x.size(0), -1)
            else:
                x = x.view(-1, self.output_conv_2 * self.conv_output_size)
                
        #Fully Connected Layers
        x = self.fc1(x)
        if Parameters.use_batch_regularization_fc:
            x = self.bn_fc1(x)
        x = self.regularization_activation_function_fc1(x)
        if self.dropout_probab_2>0: x = self.dropout2(x)
        
        if Parameters.model_complexity == "Average" or Parameters.model_complexity == "Complex":
            x = self.fc2(x)
            if Parameters.use_batch_regularization_fc:
                x = self.bn_fc2(x)
            x = self.regularization_activation_function_fc2(x)
            if self.dropout_probab_2>0: x = self.dropout2(x)
        
        x = self.fc3(x)
        #capture fc feature maps
        feature_maps_fc = x
        #print("FC Feature Maps Shape ", feature_maps.shape)
        
        return x, feature_maps_cnn, feature_maps_fc

def weights_init_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #mode=fan_out: Used for convolutional layers to account for the output size of the layer
        if Parameters.use_relu:
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity=Parameters.kaiming_uniform_nonlinearity_type, a=Parameters.kaiming_uniform_leakyrelu_a)
        else:
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity=Parameters.kaiming_uniform_nonlinearity_type, a=Parameters.kaiming_uniform_leakyrelu_a)
            #Xavier for SILU
            # nn.init.xavier_uniform_(m.weight)
            # if m.bias is not None:
            #     nn.init.constant_(m.bias, 0)
        if m.bias is not None:
            #nn.init.uniform_(m.bias, 0, 0.5)
            nn.init.constant_(m.bias, 0)
            #print(f"Convo Biases:\n{m.bias}")
            #Xavier for SILU
            # nn.init.xavier_uniform_(m.weight)
            # if m.bias is not None:
            #     nn.init.constant_(m.bias, 0)

def print_layer_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            print(f"{name}: {param.numel()} weights")

def instantiate_net(params, device):
    net = Net(params, device)
    net.to(device)
    net.parameters()
    # print("Model on CUDA",next(net.parameters()).is_cuda)
    # for param in net.parameters():
    #     print("param in CUDA",param.is_cuda)
    #print_layer_weights(net)

    return net

def Report_profiler(prof, profiler_key_averages, epoch):
    profiling_data = [
        {
            'Name': item.key,
            'CPU total %': item.cpu_time_total / sum(x.cpu_time_total for x in profiler_key_averages) *100,
            'CUDA total %': item.cuda_time_total / sum(x.cuda_time_total for x in profiler_key_averages) *100
        }
        for item in profiler_key_averages
    ]

    top_cpu = sorted(profiling_data, key=lambda x: x['CPU total %'], reverse=True)[:3]
    top_cuda = sorted(profiling_data, key=lambda x: x['CUDA total %'], reverse=True)[:3]

    # Display the results
    print("Top 3 by CPU total %:")
    for c,item in enumerate(top_cpu):
        print(f"Name: {item['Name']}, CPU total %: {item['CPU total %']:.2f}")
        if c==0: 
            cleaned_name = "top_cpu"
            val=str(re.sub(r"[^a-zA-Z0-9_\-./ ]", "_", item['Name']) + " " + str(item['CPU total %']))
            mlflow.log_param(cleaned_name,val)
            gpus = GPUtil.getGPUs()
            
    print("\nTop 3 by CUDA total %:")
    for c,item in enumerate(top_cuda):
        print(f"Name: {item['Name']}, CUDA total %: {item['CUDA total %']:.2f}")
        if c==0: 
            cleaned_name = "top_cuda"
            val=str(re.sub(r"[^a-zA-Z0-9_\-./ ]", "_", item['Name'])+" " + str(item['CUDA total %']))
            mlflow.log_param(cleaned_name,val)

    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.load * 100}% utilized")
        mlflow.log_param("gpu_utilization", gpu.load * 100)

def Train_tail_end(best_checkpoint_dict,epoch_avg_cum_loss, epoch_train_mae, epoch, eval_max_r2_epoch, best_cum_loss_epoch, best_cum_loss, best_mae, best_mae_epoch, train_loader, start_time, run_id, experiment_name,
                   net, stock_params):
    #end of training    
    print_mssg = f"End of Training: Cum loss: {epoch_avg_cum_loss:.7f}  MAE {epoch_train_mae:.7f} epoch {epoch} Best Eval R^2 epoch {eval_max_r2_epoch} Best Cum Loss: {best_cum_loss:.7f} Best MAE: {best_mae:.7f} at Best_cumloss_epoch {best_cum_loss_epoch} Best_mae_epoch {best_mae_epoch}.<p>"
    print(print_mssg)
    if Parameters.save_runs_to_md:
        helper_functions.write_to_md(print_mssg,None)

    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    
    if Parameters.enable_mlflow:
        mlflow.log_param(f"train_time", elapsed_time)

        # print("====");print(profiler.key_averages().table(sort_by="self_cpu_time_total"))
        # Report_profiler(profiler, profiler.key_averages(), epoch)

        mlflow.log_metric("epoch_avg_cum_loss",epoch_avg_cum_loss,step=epoch)
        mlflow.log_param(f"last_epoch", epoch)
                                            
    if Parameters.save_checkpoint:
        helper_functions.save_checkpoint_model(best_checkpoint_dict, best_cum_loss_epoch, eval_max_r2_epoch, best_cum_loss, epoch_avg_cum_loss, net, run_id, experiment_name, stock_params, epoch)

def set_optimizer_layer_learning_rate(net):
    conv_params = [p for name, p in net.named_parameters() if 'conv' in name]
    fc_params = [p for name, p in net.named_parameters() if 'fc' in name]
    other_params = [p for name, p in net.named_parameters() if 'conv' not in name and 'fc' not in name]

    return conv_params, fc_params, other_params


def instantiate_optimizer_and_scheduler(net, params, train_loader):
    
    conv_params, fc_params, other_params = set_optimizer_layer_learning_rate(net)

    #Optimizer init
    if Parameters.optimizer_type == "adam.Adamw":
        Parameters.optimizer = optimizer = adamw.AdamW(
            [
                {'params': conv_params, 'lr': Parameters.conv_lr if Parameters.use_layer_lr else Parameters.learning_rate},
                {'params': fc_params, 'lr': Parameters.fc_lr if Parameters.use_layer_lr else Parameters.learning_rate},
                {'params': other_params}
            ]
            , lr=Parameters.learning_rate,weight_decay=Parameters.adamw_weight_decay)
    
    if Parameters.optimizer_type == "optim.Adamw":
        Parameters.optimizer = optimizer = optim.AdamW(
            [
                {'params': conv_params, 'lr': Parameters.conv_lr if Parameters.use_layer_lr else Parameters.learning_rate},
                {'params': fc_params, 'lr': Parameters.fc_lr if Parameters.use_layer_lr else Parameters.learning_rate},
                {'params': other_params}
            ]
            , lr=Parameters.learning_rate,weight_decay=Parameters.adamw_weight_decay)
        #Parameters.optimizer = optimizer = adamw.AdamW(net.parameters(), lr=Parameters.learning_rate, weight_decay=Parameters.adamw_weight_decay)
    
    if Parameters.optimizer_type == "Adam":
        Parameters.optimizer = optimizer = optim.Adam([
                {'params': conv_params, 'lr': Parameters.conv_lr if Parameters.use_layer_lr else Parameters.learning_rate},
                {'params': fc_params, 'lr': Parameters.fc_lr if Parameters.use_layer_lr else Parameters.learning_rate},
                {'params': other_params}
            ]
            , lr=Parameters.learning_rate, weight_decay=Parameters.adam_weight_decay, betas=Parameters.adam_betas)
    
    if Parameters.optimizer_type == "SGD":
        Parameters.optimizer = optimizer = optim.SGD(
            [
                {'params': conv_params, 'lr': Parameters.conv_lr if Parameters.use_layer_lr else Parameters.learning_rate},
                {'params': fc_params, 'lr': Parameters.fc_lr if Parameters.use_layer_lr else Parameters.learning_rate},
                {'params': other_params}
            ]
            , lr=Parameters.learning_rate, momentum=Parameters.momentum_sgd)
    
    #Scheduler init
    if Parameters.scheduler_type == "OneCycleLR":
        Parameters.scheduler = scheduler = OneCycleLR(optimizer, max_lr=Parameters.oneCycleLR_max_lr, total_steps=Parameters.num_epochs_input*len(train_loader),pct_start=Parameters.oneCycleLR_pct_start)
    if Parameters.scheduler_type == "ReduceLROnPlateau":
        Parameters.scheduler = scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=Parameters.reduceLROnPlateau_mode, patience=Parameters.reduceLROnPlateau_patience, factor=Parameters.reduceLROnPlateau_factor, min_lr=Parameters.reduceLROnPlateau_min_lr, cooldown=Parameters.reduceLROnPlateau_reset_cooldown)
    if Parameters.scheduler_type == "CyclicLRWithRestarts":
        #print("batch_size cyclic",Parameters.batch_size,"epoch size",len(train_loader.dataset))
        scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=Parameters.batch_size, epoch_size=len(train_loader.dataset), restart_period=Parameters.cyclicLRWithRestarts_restart_period, t_mult=Parameters.cyclicLRWithRestarts_t_mult, policy=Parameters.cyclicLRWithRestarts_cyclic_policy, min_lr=Parameters.cyclicLRWithRestarts_min_lr, verbose=True)
        #scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=Parameters.batch_size, epoch_size=len(train_loader.dataset), restart_period=Parameters.cyclicLRWithRestarts_restart_period, t_mult=Parameters.cyclicLRWithRestarts_t_mult, policy=Parameters.cyclicLRWithRestarts_cyclic_policy, verbose=True)
    if Parameters.scheduler_type == "BayesianLR":
        scheduler = None
    if Parameters.scheduler_type == "Warmup":
        scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=10)
    if Parameters.scheduler_type == "None":
        scheduler = None
    
    return optimizer, scheduler

def summarize_epoch_statistics(net, epoch, epoch_loss, epoch_accuracy, epoch_r2, epoch_train_mae, curr_lr):
    if Parameters.nn_predict_price:
        summary_stats = {'epoch_avg_cum_loss': float(epoch_loss), 'train_r2': max(float(epoch_r2),0), 'train_mae': float(epoch_train_mae), 'curr_lr':float(curr_lr)}
    else:
        summary_stats = {'epoch_avg_cum_loss': float(epoch_loss), 'train_accuracy': float(epoch_accuracy), 'curr_lr':float(curr_lr)}
    
    # Iterate through model layers to collect weight and gradient statistics
    if Parameters.log_weights:
        for name, param in net.named_parameters():
            if param.requires_grad:
                # Collect weight statistics
                #print("###weights",param.data.mean().item())
                summary_stats[f"{name}_weight_mean"] = float(param.data.mean().item())
                summary_stats[f"{name}_weight_std"] = float(param.data.std().item()) if param.data.numel() > 1 else 0.0
                summary_stats[f"{name}_weight_min"] = float(param.data.min().item())
                summary_stats[f"{name}_weight_max"] = float(param.data.max().item())
                
                # Collect gradient statistics
                if param.grad is not None:
                    #print(f"###grad {name}",param.grad.mean().item())
                    summary_stats[f"{name}_grad_mean"] = float(param.grad.mean().item()) if param.grad is not None else 0.0
                    summary_stats[f"{name}_grad_std"] = float(param.grad.std().item()) if param.grad.numel() > 1 else 0.0
                    summary_stats[f"{name}_grad_min"] = float(param.grad.min().item()) if param.grad is not None else 0.0
                    summary_stats[f"{name}_grad_max"] = float(param.grad.max().item()) if param.grad is not None else 0.0

    if Parameters.enable_mlflow:
        mlflow.log_metrics(summary_stats,step=epoch)

    return summary_stats

def Train(params, train_loader, train_gaf_image_dataset_list_f32, train_stocks_dataset_df, net, run, run_id, experiment_name, device, stock_params):

    #enable grad
    torch.set_grad_enabled(True)

    best_checkpt_dict = {
            'run_id': None,
            'epoch': None,
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'loss': None,
            }

    inputs_list = []

    best_avg_cum_loss_epoch = torch.tensor(0.0, device=device, dtype=torch.float32)
    best_avg_cum_loss = Parameters.min_best_cum_loss
    best_avg_checkpoint_cum_loss = Parameters.best_avg_checkpoint_cum_loss
    best_train_mae = torch.tensor(100.0, device=device, dtype=torch.int16)
    best_train_mae_epoch = torch.tensor(0.0, device=device, dtype=torch.float32)

    #analysis init
    num_classes = 2 if Parameters.nn_predict_price == 0 else None
    task_type = "binary" if Parameters.nn_predict_price == 0 and num_classes == 2 else "multiclass"
    if Parameters.nn_predict_price == 0: # Classification
        acc_metric = Accuracy(task=task_type, num_classes=num_classes).to(device)
        r2_metric = None
    else: # Regression
        r2_metric = R2Score().to(device)
        acc_metric = None

    train_mae_metric = MeanAbsoluteError().to(device)

    #torch.set_printoptions(threshold=torch.inf)

    start_time = time.time()
    ReduceLROnPlateau_blocked = 0

    print_mssg=f"Train params: learning_rate: {params.learning_rate}, momentum:{Parameters.momentum_sgd} loss_threshold {params.loss_stop_threshold}<p>"
    #print(print_mssg)
    if Parameters.save_runs_to_md:
        helper_functions.write_to_md(print_mssg,None)

    #don't init to fine tune
    if Parameters.train and not Parameters.fine_tune:
        net.apply(weights_init_he)

    criterion = Parameters.function_loss

    #scale loss as I used mixed precision
    if Parameters.use_mixed_precision:
        gradScaler = torch.cuda.amp.GradScaler()

    optimizer, scheduler = instantiate_optimizer_and_scheduler(net, params, train_loader)

    #initial load of the model
    if Parameters.fine_tune:
        helper_functions.load_checkpoint_model(net, device, stock_params, train_loader, "fine_tune")
    else:
        model_checkpoint_fname_with_dir = f'{Parameters.checkpoint_dir}/{Parameters.model_checkpoint_fname}_{stock_params.train_stock_tickers}_{stock_params.eval_stock_tickers}_{Parameters.model_uuid}.pth'
        if len(stock_params.train_stocks) > 1 and os.path.exists(model_checkpoint_fname_with_dir):
            helper_functions.load_checkpoint_model(net, device, stock_params, train_loader, "pre-train")

        print(f"Loaded weight for 'conv2.weight': {net.state_dict()['conv2.weight'][0][0]}")

            # print(f"Layer: {name}")
            # print(f"First 5 weights: {param.data.view(-1)[:5]}")
            # break  # Print only the first layer's weights

    bayesian_LR_warmup_loss_list = []
   
    # profiler = torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
    # profiler.start()
    for epoch in range(params.num_epochs_input):

        print(f"\033[32mStart Train Epoch [{epoch + 1}/{Parameters.num_epochs_input}]\033[0m")

        if Parameters.scheduler_type == "CyclicLRWithRestarts":
            scheduler.step()

        #warm up LR rate for bayesian
        if Parameters.scheduler_type == "BayesianLR" and (Parameters.bayesian_warmup_epochs-1 > epoch):
            for param_group in optimizer.param_groups:
                param_group['lr'] = Parameters.bayesian_warmup_learning_rates[epoch]
            
        #init bayesianLR scheduler
        if Parameters.scheduler_type == "BayesianLR" and Parameters.bayesian_warmup_epochs == epoch:
            bayesian_LR_warmup_loss_list_np = np.array(bayesian_LR_warmup_loss_list).reshape(-1, 1)
            learning_rates_np = np.array(Parameters.bayesian_warmup_learning_rates).reshape(-1, 1)
            
            Parameters.scheduler = scheduler = GPyOpt.methods.BayesianOptimization(f=None, 
                                                initial_design_numdata=len(bayesian_LR_warmup_loss_list), 
                                                X=learning_rates_np, 
                                                Y=bayesian_LR_warmup_loss_list_np,
                                                domain=Parameters.bayesianLR_bounds,
                                                acquisition_type='LCB', acquisition_weight=2,
                                                xi=0.1,
                                                batch_size=Parameters.batch_size)
    
        #get only the last epochs for analysis
        feature_maps_cnn_list =[]
        feature_maps_fc_list =[]
        epoch_accuracy = 0.0
        epoch_r2 = 0.0
        epoch_total_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        epoch_avg_cum_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        total_samples = 0
        
        #weights/grad analysis
        #epoch_gradients = {name: [] for name, param in net.named_parameters() if param.requires_grad}

        if Parameters.nn_predict_price == 0:
            acc_metric.reset()
        else:
            r2_metric.reset()
        
        train_mae_metric.reset()

        gradients_dict = {name: [] for name, _ in net.named_parameters() if 'weight' in name}

        weights_dict = {name: [] for name, _ in net.named_parameters() if 'weight' in name}

        curr_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR {curr_lr:.6f}")

        for i, data in enumerate(train_loader, 0):
            #print(f"Batch {i}, len trianloader {len(train_loader)} len trainloaderdata {len(train_loader.dataset)}")

            #non_blocking combined with dataloader pin_memory for simultaneous transfer&computation
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            #print("labels",labels[0])
            if Parameters.nn_predict_price==False:
                labels = labels.flatten().long()

            # zero the parameter gradients
            optimizer.zero_grad()

            #mixed precision
            autocast_context = torch.autocast(device_type='cuda', dtype=torch.float16) if (Parameters.use_mixed_precision) else contextlib.nullcontext()
            if epoch == 0:
                    input_tensor = inputs.data
                    inputs_list.append(input_tensor)
                    
            if Parameters.save_arch_bool:
                helper_functions.Save_Model_Arch(net, run_id, inputs.shape, [inputs.dtype], "train", experiment_name)
                Parameters.save_arch_bool = False
            
            with autocast_context:
                # forward + backward + optimize
                #print("outputs shape",inputs.shape,inputs)
                outputs, feature_maps_cnn, feature_maps_fc = net(inputs)
                #print("outputs shape",outputs.shape,outputs)
                #print("labels",labels)
                loss = criterion(outputs, labels)
                #print("LOSS",loss, "labda",Parameters.fc_lambda_ssim, "1-ssim",(1 - Parameters.fc_ssim_score))
                #print("Standard LOSS",loss)
                if Parameters.use_ssim_adjusted_loss:
                    loss = loss + Parameters.lambda_ssim*Parameters.cnn_fc_lambda_ssim_ratio*(1 - Parameters.cnn_ssim_score) + Parameters.lambda_ssim*(1-Parameters.cnn_fc_lambda_ssim_ratio)*(1 - Parameters.fc_ssim_score)
                    
            feature_maps_cnn_list.append(feature_maps_cnn)
            feature_maps_fc_list.append(feature_maps_fc)
            if i==0 and epoch==0:
                if Parameters.enable_mlflow:
                    input_np = inputs.detach().cpu().numpy()
                    model_signature = mlflow.models.infer_signature(input_np,outputs.detach().cpu().numpy())
            
            # For classification accuracy
            if Parameters.nn_predict_price == 0:
                acc_metric.update(outputs, labels)
            else:
                r2_metric.update(outputs, labels)

            train_mae_metric.update(outputs, labels)

            if loss is not None:
                                
                #scale loss due to mixed precision
                if Parameters.use_mixed_precision:
                    gradScaler.scale(loss).backward()
                    gradScaler.unscale_(optimizer)
                    if Parameters.use_clip_grad_norm:
                        total_norm_before = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=Parameters.grad_norm_clip_max)
                        if total_norm_before > Parameters.grad_norm_clip_max:
                            print("****[WARNING Clipping Applied]*****",total_norm_before,"epoch", epoch)
                            if Parameters.enable_mlflow:
                                mlflow.log_metric("norm_clip_on",0.5,step=epoch)
                    gradScaler.step(optimizer)
                    gradScaler.update()
                else:
                    loss.backward()
                    if Parameters.use_clip_grad_norm:
                        total_norm_before = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=Parameters.grad_norm_clip_max)
                        if total_norm_before > Parameters.grad_norm_clip_max:
                            print("****[WARNING Clipping Applied]*****",total_norm_before,"epoch", epoch)
                            if Parameters.enable_mlflow:
                                mlflow.log_metric("norm_clip_on",0.5,step=epoch)
                    optimizer.step()

                if Parameters.scheduler_type == "OneCycleLR":
                    scheduler.step()

                if Parameters.scheduler_type == "CyclicLRWithRestarts":
                    scheduler_param_group = scheduler.batch_step()
                    #print("Getting Learning Rate ", scheduler_param_group['lr'])
                    #print("Getting weight decay", scheduler_param_group['weight_decay'])
                    #print("In Batch Loss", loss.detach().item())
                
                # Print optimizer's state_dict
                # print("Optimizer's state_dict:")
                # for var_name in optimizer.state_dict():
                #     print(var_name, "\t", optimizer.state_dict()[var_name])

                #average loss for batches
                batch_size = inputs.size(0)
                unscaled_loss = loss.detach()
                batch_loss = unscaled_loss.item() * batch_size
                epoch_total_loss += batch_loss
                total_samples += batch_size
                
                # if first_batch and ((epoch + 1) % params.epoch_running_gradients_check == 0):
                #     for name, param in net.named_parameters():
                #         if 'weight' in name:
                #             if param.grad is not None:
                #                 gradients_dict[name].append(param.grad.detach().clone().cpu().numpy())
                #             weights_dict[name].append(param.detach().clone().cpu().numpy())

                    #plot_data.plot_weights_gradients(weights_dict, gradients_dict, epoch,experiment_name, run_id)

                #first_batch = False

            # torch.cuda.empty_cache()
            # _ = gc.collect() #force garbage collect

        epoch_avg_cum_loss = epoch_total_loss / total_samples

        if Parameters.scheduler_type == "OneCycleLR":
            if i==0: print(f"Batch 0 epoch {epoch} Current OneCycleLR Learning Rate:",scheduler.get_last_lr(), "epoch avg cum loss",epoch_avg_cum_loss.item())

        if Parameters.scheduler_type == "BayesianLR" and epoch < Parameters.bayesian_warmup_epochs:
            bayesian_LR_warmup_loss_list.append(epoch_avg_cum_loss.cpu().item())
            print(f"Collected with LR {Parameters.bayesian_warmup_learning_rates[epoch]} Loss {epoch_avg_cum_loss}")
        
        epoch_train_mae = train_mae_metric.compute()
        if epoch_train_mae < best_train_mae:
            best_train_mae = epoch_train_mae
            best_train_mae_epoch = epoch
            if Parameters.enable_mlflow:
                mlflow.log_metric("best_train_mae",best_train_mae,step=epoch)
        
        # store inputs for metrics calculations e.g. correl
        if epoch == 0:
            stack_input = torch.stack(inputs_list, dim=0)

        # report curr cum loss
        if ((epoch + 1) % Parameters.epoch_running_loss_check == 0):
                print_mssg = f"[{(epoch + 1):d}, {(i + 1):5d}] Cum loss: {(epoch_avg_cum_loss):.7f} MAE {epoch_train_mae}<p>"
                print(print_mssg) 
                if Parameters.save_runs_to_md:
                    helper_functions.write_to_md(print_mssg,None)

        # best_cum_loss improved
        if epoch_avg_cum_loss < best_avg_cum_loss:
            best_avg_cum_loss_epoch = epoch
            best_avg_cum_loss = epoch_avg_cum_loss
            if Parameters.enable_mlflow:
                mlflow.log_metric("best_avg_cum_loss",best_avg_cum_loss,step=epoch)
        
        # loss<threshold: update checkpoint
        print(f"BEFORE Updating Checkpoint {epoch} loss {epoch_avg_cum_loss} and best loss {best_avg_checkpoint_cum_loss}")
        if epoch_avg_cum_loss < best_avg_checkpoint_cum_loss or epoch==0:
            best_avg_checkpoint_cum_loss = epoch_avg_cum_loss
            best_avg_cum_loss_epoch = epoch
            best_avg_cum_loss = epoch_avg_cum_loss

        if epoch==0 and not helper_functions.check_if_checkpoint_exists(stock_params):
            best_checkpt_dict=helper_functions.update_best_checkpoint_dict(best_avg_cum_loss_epoch, Parameters.eval_max_r2_epoch, run_id, net.state_dict(), Parameters.optimizer.state_dict(), epoch_avg_cum_loss)
            if Parameters.save_checkpoint:
                helper_functions.save_checkpoint_model(best_checkpt_dict,best_avg_cum_loss_epoch, Parameters.eval_max_r2_epoch, best_avg_cum_loss, epoch_avg_cum_loss, net, run_id, experiment_name, stock_params, epoch)
        
        #Eval
        #evaluation test image generation
        if epoch == 0:
            eval_gaf_image_dataset_list_f32, test_loader, actual_tensor, eval_stock_dataset_df = pipeline.generate_evaluation_images(stock_params, run, experiment_name, device)
            #DTW multiple train stocks series
            if stock_params.train_count > 1:
                plot_data.calc_merged_series_dtw_distance(train_stocks_dataset_df,eval_stock_dataset_df)
            #SSIM
            if len(train_gaf_image_dataset_list_f32) == len(eval_gaf_image_dataset_list_f32):
                mse=F.mse_loss(torch.from_numpy(train_gaf_image_dataset_list_f32), torch.from_numpy(eval_gaf_image_dataset_list_f32)).item()
                pipeline.report_image_similarities_eval(stock_params,train_gaf_image_dataset_list_f32, eval_gaf_image_dataset_list_f32, epoch)
                if Parameters.enable_mlflow:
                    mlflow.log_metric("input_imgs_MSE",mse,step=epoch)

        if (epoch + 1) % Parameters.eval_at_epoch_multiple == 0:
            print("=====Evaluate at Training======")
            #load best checkpoint
            #net_eval = pipeline.load_checkpoint_for_eval(device, stock_params, train_loader)
            net  = set_model_for_eval(net)
            torch.set_grad_enabled(False)
            
            eval_error_stats = pipeline.evaluate_and_report(net, stock_params, device, test_loader, run, run_id, experiment_name, 
                                train_gaf_image_dataset_list_f32, eval_gaf_image_dataset_list_f32,
                                train_stocks_dataset_df,
                                feature_maps_cnn_list, feature_maps_fc_list, stack_input, epoch)
            
            print(f"Compare {eval_error_stats['eval_R2']} and {Parameters.eval_max_r2}")
            if eval_error_stats['eval_R2'] > Parameters.eval_max_r2:
                Parameters.eval_max_r2 = eval_error_stats['eval_R2']
                Parameters.eval_max_r2_epoch = epoch
                if Parameters.enable_mlflow:
                    mlflow.log_metric("max_eval_R2",Parameters.eval_max_r2, epoch)
                
                best_checkpt_dict=helper_functions.update_best_checkpoint_dict(best_avg_cum_loss_epoch, Parameters.eval_max_r2_epoch, run_id, net.state_dict(), Parameters.optimizer.state_dict(), epoch_avg_cum_loss)
                
                helper_functions.save_feature_maps(feature_maps_cnn_list, feature_maps_fc_list)

                #print("NOW train",feature_maps_fc_list[0],"len",len(feature_maps_fc_list),"shape",feature_maps_fc_list[0].shape)
            
            print(f"\033[32mMax Eval R^2 {Parameters.eval_max_r2} at epoch {Parameters.eval_max_r2_epoch}\033[0m")
            
            if ((epoch+1) % Parameters.save_checkpoint_at_epoch_multiple == 0) and Parameters.save_checkpoint:
                helper_functions.save_checkpoint_model(best_checkpt_dict,best_avg_cum_loss_epoch, Parameters.eval_max_r2_epoch, best_avg_cum_loss, epoch_avg_cum_loss, net, run_id, experiment_name, stock_params, epoch)

            #return train and grad enabled post eval
            net.train()
            torch.set_grad_enabled(True)
            
        #report for during training analytics on this epoch
        #calc accur/r^2
        if (epoch + 1) % Parameters.log_params_at_epoch_multiple == 0:
            if Parameters.nn_predict_price == 0:
                epoch_accuracy = acc_metric.compute()
                mssg=f"End Train Epoch [{epoch + 1}/{Parameters.num_epochs_input}], Average Loss: {epoch_avg_cum_loss:.6f}, Train MAE: {epoch_train_mae:.6f}, Accuracy: {epoch_accuracy:.4f}"
            else:
                epoch_r2 = r2_metric.compute() if Parameters.nn_predict_price else None
                mssg = f"\033[32mEnd Train Epoch [{epoch + 1}/{Parameters.num_epochs_input}], Average Loss: {epoch_avg_cum_loss:.6f}, Train MAE: {epoch_train_mae:.6f}, Train R^2: {epoch_r2:.4f}\033[0m"
                print(mssg)
                #helper_functions.write_scenario_to_log_file(mssg)
                epoch_stats = summarize_epoch_statistics(net, epoch, epoch_avg_cum_loss, epoch_accuracy,epoch_r2, epoch_train_mae, curr_lr)
                #helper_functions.write_scenario_to_log_file(epoch_stats)

        # if Parameters.log_params_at_epoch_multiple!=1:
        #     print(f"Train Epoch [{epoch + 1}/{Parameters.num_epochs_input}], Cum loss: {epoch_avg_cum_loss:.7f}  Train MAE {epoch_train_mae:.7f} Train R^2 {epoch_r2:.7f} Best Avg Cum Loss: {best_avg_cum_loss:.7f} Best Train MAE: {best_train_mae:.7f} at Best_cumloss_epoch {best_avg_cum_loss_epoch} Best_mae_epoch {best_train_mae_epoch}")

        #exit if below loss threshold
        #print(f"Comparing Stop - Cum Loss {epoch_cum_loss} Vs {params.loss_stop_threshold}")
        #TODO create function
        if epoch_avg_cum_loss < Parameters.loss_stop_threshold:
            print(f"[WARNING] Epoch Cum Loss is less than {params.loss_stop_threshold}:{epoch_avg_cum_loss} at {epoch}. MAE: {epoch_train_mae:.6f}. Stopping training.")
            
            if Parameters.save_runs_to_md:
                helper_functions.write_to_md(f"Epoch Cum Loss is less than {params.loss_stop_threshold}:{epoch_avg_cum_loss} at {epoch}. MAE: {epoch_train_mae:.6f}. Stopping training.<p>",None)
            
            if Parameters.nn_predict_price == 0:
                epoch_accuracy = acc_metric.compute()
                mssg=f"Epoch [{epoch + 1}/{params.num_epochs_input}], Average Loss: {epoch_avg_cum_loss:.6f}, MAE: {epoch_train_mae:.6f}, Accuracy: {epoch_accuracy:.4f}"
            else:
                epoch_r2 = r2_metric.compute() if Parameters.nn_predict_price else None
                mssg = f"Epoch [{epoch + 1}/{params.num_epochs_input}], Average Loss: {epoch_avg_cum_loss:.6f}, MAE: {epoch_train_mae:.6f}, R^2: {epoch_r2:.4f}"
                
                print(mssg)
                helper_functions.write_scenario_to_log_file(mssg)
                epoch_stats = summarize_epoch_statistics(net, epoch, epoch_avg_cum_loss, epoch_accuracy,epoch_r2, epoch_train_mae, curr_lr)
                helper_functions.write_scenario_to_log_file(epoch_stats)

            #profiler.stop()

            Train_tail_end(best_checkpt_dict, epoch_avg_cum_loss, epoch_train_mae, epoch, Parameters.eval_max_r2_epoch, best_avg_cum_loss_epoch, best_avg_cum_loss, best_train_mae, best_train_mae_epoch, train_loader, start_time, run_id, experiment_name, net, stock_params)

            #return net, model_signature, stack_input
            return net, model_signature if 'model_signature' in locals() else None or None, stack_input, feature_maps_cnn_list, feature_maps_fc_list
        
        #track scheduler auto step down
        if Parameters.scheduler_type == "ReduceLROnPlateau":
            #scheduler step LR
            prior_lr = scheduler.optimizer.param_groups[0]['lr']
            
            scheduler.step(epoch_avg_cum_loss.item())
            #print("LR BEFORE STEP",prior_lr,"LR AFTER",scheduler.optimizer.param_groups[0]['lr'])
            #print(f"Comparing Blocked {ReduceLROnPlateau_blocked} VS Cooldown {Parameters.reduceLROnPlateau_reset_cooldown}")
            if Parameters.reduceLROnPlateau_enable_reset:
                if ReduceLROnPlateau_blocked < Parameters.reduceLROnPlateau_reset_cooldown:
                    
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = Parameters.reduceLROnPlateau_reset_rate
                    print("I've reset rate at BLCOKED to ",Parameters.reduceLROnPlateau_reset_rate)
                    ReduceLROnPlateau_blocked+=1

                    current_lr = scheduler.optimizer.param_groups[0]['lr']
                    print(f"prior LR",prior_lr,"curr",current_lr)
                    
                    # If the scheduler reduced the learning rate, update the last best epoch
                    if prior_lr > current_lr:
                        print(f"[INFO] ReduceLROnPlateau adjusted LR from {prior_lr} to {current_lr}")

        # LR manual reset logic
        if Parameters.reduceLROnPlateau_enable_reset:
            if Parameters.scheduler_type == "ReduceLROnPlateau":
                print("curr LR",scheduler.optimizer.param_groups[0]['lr'], " VS ",(Parameters.reduceLROnPlateau_min_lr + Parameters.reduceLROnPlateau_min_lr * 1e-2 ))
                if scheduler.optimizer.param_groups[0]['lr'] < (Parameters.reduceLROnPlateau_min_lr + Parameters.reduceLROnPlateau_min_lr * 1e-2 ):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = Parameters.reduceLROnPlateau_reset_rate
                        ReduceLROnPlateau_blocked = 0
                    print(f"[WARNING] LR reset to {Parameters.reduceLROnPlateau_reset_rate} due to hitting min loss at epoch {epoch}")

        if Parameters.train_max_r2 < epoch_r2:
            Parameters.train_max_r2 = epoch_r2
            Parameters.train_max_r2_epoch = epoch
            if Parameters.enable_mlflow:
                    mlflow.log_metric("max_train_R2",Parameters.train_max_r2, epoch)
        
        print(f"\033[32mMax Train R^2 {Parameters.train_max_r2} at epoch {Parameters.train_max_r2_epoch}\033[0m")

        if Parameters.scheduler_type == "BayesianLR" and Parameters.bayesian_warmup_epochs <= epoch and ((epoch + 1) % Parameters.bayes_find_lr_frequency_epochs == 0):
            #add observations to bayesian
            suggested_lr = scheduler.suggest_next_locations()[0][0]
            scheduler.X = np.vstack((scheduler.X, np.array([[suggested_lr]])))
            scheduler.Y = np.vstack((scheduler.Y, np.array([[epoch_avg_cum_loss.item()]])))
            scheduler._update_model()

            #update optimizer lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = suggested_lr

            #bayesian_LR_warmup_loss_list.append(suggested_lr.item())
            bayesian_LR_warmup_loss_list.append(epoch_avg_cum_loss.cpu().item())
            Parameters.bayesian_warmup_learning_rates.append(suggested_lr.item())

            if Parameters.enable_mlflow:
                if epoch_avg_cum_loss.item() < Parameters.bayes_loss_threshold_to_log:
                    bayesian_lr_loss_mlflow = {
                        "bayes_lr": suggested_lr.item(),
                        "bayes_loss": epoch_avg_cum_loss.cpu().item()
                    }
                    
                    if Parameters.enable_mlflow:
                        mlflow.log_metrics(bayesian_lr_loss_mlflow, epoch)

            plot_data.plot_lr_vs_loss(Parameters.bayesian_warmup_learning_rates, bayesian_LR_warmup_loss_list)
        
    #end training without reaching loss_threshold
    #profiler.stop()

    Train_tail_end(best_checkpt_dict ,epoch_avg_cum_loss, epoch_train_mae, epoch, Parameters.eval_max_r2_epoch, best_avg_cum_loss_epoch, best_avg_cum_loss, best_train_mae, best_train_mae_epoch, train_loader, start_time, run_id, experiment_name, net, stock_params)
    
    torch.set_printoptions()

    return net, model_signature if 'model_signature' in locals() else None, stack_input, feature_maps_cnn_list, feature_maps_fc_list

def Test(test_loader, net, stock_ticker, epoch, device, experiment_name, run):
    inputs_list = []
    accuracy = 0
    correct_2dp_list = []
    correct_1dp_list = []
    correct_2dp_score = torch.tensor(0.0, device=device, dtype=torch.float32)
    correct_1dp_score = torch.tensor(0.0, device=device, dtype=torch.float32)
    correct_classification_score = torch.tensor(0, device=device, dtype=torch.int)
    correct_classification_list = []
    error_list = []
    #dataframe vars
    predicted_list = []
    actual_list = []
    abs_percentage_diffs_list = []
    stack_input=None
    stack_predicted=None
    batch_abs_percentage_diff = torch.tensor(0.0, device=device, dtype=torch.float32)

    feature_maps_cnn_list =[]
    feature_maps_fc_list = []

    torch.set_printoptions(threshold=torch.inf)

    # Regression
    r2_metric = R2Score().to(device)
    r2_metric.reset()

    for i, data in enumerate(test_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        # if i==0: 
        #     print("TEST inputs",inputs[0][0])
        #     print("TEST labels",labels)

        #mixed precision
        autocast_context = torch.autocast(device_type='cuda', dtype=torch.float16) if (Parameters.use_mixed_precision) else contextlib.nullcontext()

        with autocast_context:
            with torch.no_grad():
                outputs, feature_maps_cnn, feature_maps_fc = net(inputs)
                # if i==0: 
                #     print("TEST outputs",outputs)
        feature_maps_cnn_list.append(feature_maps_cnn)
        feature_maps_fc_list.append(feature_maps_fc)
        #print("outputs",outputs)

        r2_metric.update(outputs, labels)
        
        # for name, param in net.named_parameters():
        #     if param.requires_grad:
                # Collect weight statistics
                #print("###weights",param.data.mean().item())
                #print(f"{name}_weight_mean {param.data.mean().item()}")
                # summary_stats[f"{name}_weight_std"] = float(param.data.std().item()) if param.data.numel() > 1 else 0.0
                # summary_stats[f"{name}_weight_min"] = float(param.data.min().item())
                # summary_stats[f"{name}_weight_max"] = float(param.data.max().item())

        input_tensor = inputs.data
        actual_tensor = labels.data
        #print(f"Actual {i}",actual_tensor[:1])
        if Parameters.nn_predict_price:
            predicted_tensor = outputs.data
        else:
            _, predicted_tensor = torch.max(outputs, 1)
        
        inputs_list.append(input_tensor)

        predicted_list.append(predicted_tensor)

        actual_list.append(actual_tensor)
        
        #calculate batch pct difference
        batch_abs_percentage_diff = (torch.abs(predicted_tensor - actual_tensor) / torch.abs(actual_tensor)) * 100
        abs_percentage_diffs_list.extend(batch_abs_percentage_diff.detach().cpu().numpy().flatten())

        #Check IQR
        error_tensor = (predicted_tensor - actual_tensor)
        error_list.append(error_tensor)
        
        #accuracy
        if Parameters.nn_predict_price:
            correct_2dp_tensor = (torch.abs(predicted_tensor - actual_tensor)<= 0.01)
            correct_2dp_list.append(correct_2dp_tensor)
            
            #print("predicted",predicted_tensor,"actual",actual_tensor)
            correct_1dp_tensor = (torch.abs(predicted_tensor - actual_tensor)<= 0.1)
            correct_1dp_list.append(correct_1dp_tensor)
        else:
            correct_classification_tensor = torch.eq(predicted_tensor, actual_tensor)
            correct_classification_list.append(correct_classification_tensor)
            #print("list",correct_classification_tensor)

    torch.set_printoptions()

    if Parameters.nn_predict_price:
        correct_2dp_score = calculate_score(correct_2dp_list)
        correct_1dp_score = calculate_score(correct_1dp_list)
    else:
        correct_classification_score = calculate_score(correct_classification_list)
        if Parameters.enable_mlflow:
            confusion_mtx(actual_list,predicted_list, stock_ticker, experiment_name, run.info.run_id)
        else:
            confusion_mtx(actual_list,predicted_list, stock_ticker, None, None)

    error_list_iqr, error_pct_outside_iqr = compute_stats.calculate_iqr(error_list)

    text_mssg=f"error_pct_outside_iqr {error_pct_outside_iqr}<p>"
    if Parameters.save_runs_to_md:
        helper_functions.write_to_md(text_mssg,None)

    stack_input = torch.stack(inputs_list, dim=0)
    
    #actual and predicted mean
    stack_actual = torch.stack(actual_list, dim=0).flatten()

    stack_predicted = torch.stack(predicted_list, dim=0).flatten()
        
    accuracy = [correct_2dp_score, correct_1dp_score, correct_classification_score]

    if Parameters.max_acc_1dp < correct_1dp_score:
        Parameters.max_acc_1dp = correct_1dp_score
        Parameters.max_acc_1dp_epoch = epoch

    print(); print(f"\033[32mStock {stock_ticker} Accuracy 2 decimal places: {correct_2dp_score}, "
            f"Accuracy 1 decimal places: {correct_1dp_score}, Max Accuracy 1 dp {Parameters.max_acc_1dp} at epoch {Parameters.max_acc_1dp_epoch} "
            f"; Accuracy Classification {correct_classification_score/100} cum_loss\033[0m\n")

    text_mssg=(f"Stock {stock_ticker} Accuracy 2 decimal places: {correct_2dp_score} <p> \
               Accuracy 1 decimal places: {correct_1dp_score} , Max Accuracy 1 dp {Parameters.max_acc_1dp} at epoch {Parameters.max_acc_1dp_epoch} <p> \
               ; Accuracy Classification {correct_classification_score/100} <p>")

    if Parameters.save_runs_to_md:
        helper_functions.write_to_md(text_mssg,None)

    epoch_r2 = r2_metric.compute() if Parameters.nn_predict_price else None
    mssg = f"Eval Torch R^2: {epoch_r2:.4f}"
    print(mssg)

    if Parameters.best_eval_r2 < epoch_r2:
        Parameters.best_eval_r2 = epoch_r2
        Parameters.best_eval_r2_epoch = epoch

    # log metrics
    if Parameters.enable_mlflow:
        if Parameters.nn_predict_price:
            accuracy_metrics = {f"accuracy_2dp_score": (correct_2dp_score.double().item())/100,
                    f"accuracy_1dp_score": (correct_1dp_score.double().item())/100,
                    f"max_accuracy_1dp_score": (Parameters.max_acc_1dp.double().item())/100,
                    f"error_pct_outside_iqr": error_pct_outside_iqr,
                    f"eval_r2_torch": float(epoch_r2),
                    f"best_eval_r2_torch": float(Parameters.best_eval_r2),
                    f"best_eval_r2_torch_epoch": float(Parameters.best_eval_r2_epoch)}
        else:
            accuracy_metrics = {f"accuracy_classification_score": correct_classification_score.double().item()}
        
        mlflow.log_metrics(accuracy_metrics, step=epoch)

    return stack_input, predicted_list, actual_list, accuracy, stack_actual, stack_predicted, feature_maps_cnn_list, feature_maps_fc_list,

def confusion_mtx(true_labels, predicted_labels, stock_ticker, experiment_name, run_id):
    flattened_true_labels = flatten_labels(true_labels)
    flattened_predicted_labels = flatten_labels(predicted_labels)
    cm = confusion_matrix(flattened_true_labels, flattened_predicted_labels)
    plot_data.plot_confusion_matrix(cm, stock_ticker, True, experiment_name, run_id)

def flatten_labels(label_list):
    flattened_labels = []
    for label in label_list:
        if label.dim() == 2:
            flattened_labels.extend(label.view(-1).cpu().numpy())
        else:
            flattened_labels.extend(label.cpu().numpy())
    return np.array(flattened_labels)

def calculate_score(tensor_list):
    if Parameters.nn_predict_price:
        stack = torch.stack(tensor_list, dim=1).flatten()
        return (torch.sum(stack)/len(stack))*100
    else:
        stack = torch.stack(tensor_list, dim=1).flatten()
        #print("*****Classification sum",stack.sum(),"total",len(stack))
        return (stack.sum()/len(stack))*100

def set_model_for_eval(net):
    net.eval()
    return net