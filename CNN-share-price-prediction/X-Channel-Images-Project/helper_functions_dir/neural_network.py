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

import numpy as np
from sklearn.metrics import confusion_matrix

import mlflow

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import adamw as adamw
import cyclic_scheduler as cyclic_scheduler 
import plot_data as plot_data
import image_transform as image_transform
import helper_functions as helper_functions
import compute_stats as compute_stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parameters import Parameters

from torch.profiler import profile, ProfilerActivity

# #set gpu env
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("CUDA Available:", torch.cuda.is_available())
# print("Num GPUs available",torch.cuda.device_count())
# print("At training - Device",device)
# print("cuda version",torch.version.cuda)
# #track mem used by tensors
# torch.cuda.memory_allocated()
#track cached mem used by tensors
#torch.cuda.memory_cached()
  
class Net(nn.Module):
    def __init__(self, params, device):
        super(Net, self).__init__()
        
        if params.model_name:
            self.name = params.model_name
        self.totalparams = 0
        self.output_conv_2= params.output_conv_2
        self.conv_output_size=0
        self.dropout_probab = params.dropout_probab
        #print();print("Convos & dropoutP:", params.output_conv_1, params.output_conv_2, params.dropout_probab)
        
        #num channels input, num channels output, filter size
        self.conv1 = nn.Conv2d(1, params.output_conv_1, params.filter_size_1, params.stride_1)
        self.bn1 = nn.BatchNorm2d(params.output_conv_1)
        self.regularization_activation_function_1 = Parameters.regularization_function
        #self.relu1 = nn.ReLU() 
        #filtersize,stride.
        #maxpool acts the same way in each channel, so doesn't need to be fed the num channels of the input
        self.pool = nn.MaxPool2d(kernel_size=params.filter_size_2, stride = params.stride_2)
        self.conv2 = nn.Conv2d(params.output_conv_1, params.output_conv_2, params.filter_size_3,params.stride_1)
        self.bn2 = nn.BatchNorm2d(params.output_conv_2)
        self.regularization_activation_function_2 = Parameters.regularization_function
        #self.relu2 = nn.ReLU() 

        H_out_1, W_out_1 = image_transform.conv_output_shape_dynamic((params.image_resolution_y, params.image_resolution_x), kernel_size=params.filter_size_1,stride=params.stride_1)
        H_out_2, W_out_2 = image_transform.conv_output_shape_dynamic((H_out_1, W_out_1), kernel_size=params.filter_size_2,stride=params.stride_2)
        H_out_3, W_out_3 = image_transform.conv_output_shape_dynamic((H_out_2, W_out_2), kernel_size=params.filter_size_3, stride=params.stride_1)
        H_out_4, W_out_4 = image_transform.conv_output_shape_dynamic((H_out_3, W_out_3), kernel_size=params.filter_size_2, stride = params.stride_2)
        
        # print("imgres", params.image_resolution_x, params.image_resolution_y)
        # print("H_out_1, W_out_1",H_out_1, W_out_1)
        # print("H_out_2, W_out_2",H_out_2, W_out_2)
        # print("H_out_3, W_out_4",H_out_3, W_out_3)
        # print("H_out_4, W_out_4",H_out_4, W_out_4)
        # print("outputconv2", params.output_conv_2)
        self.conv_output_size = H_out_4 * W_out_4

        self.fc1 = nn.Linear(params.output_conv_2 * self.conv_output_size, params.output_FC_1)
        self.bn_fc1 = nn.BatchNorm1d(params.output_FC_1)
        self.regularization_activation_function_fc1 = Parameters.regularization_function
        #self.relu_fc1 = nn.ReLU()  
        self.fc2 = nn.Linear(params.output_FC_1, params.output_FC_2)
        self.bn_fc2 = nn.BatchNorm1d(params.output_FC_2)
        self.regularization_activation_function_fc2 = Parameters.regularization_function
        #self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(params.output_FC_2, params.final_FCLayer_outputs)
        
        self.dropout1 = nn.Dropout(params.dropout_probab)
        self.dropout2 = nn.Dropout(params.dropout_probab)

        # compute the total number of parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #print(self.name + ': total params:', total_params)
        self.totalparams=total_params

    def forward(self, x):
        #BatchNorm after Conv and before Pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.regularization_activation_function_1(x)
        #x = self.relu1(x)
        x = self.pool(x)
        #BatchNorm after Conv and before Pooling
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.regularization_activation_function_2(x)
        #x = self.relu2(x)
        x = self.pool(x)
        x = x.view(-1, self.output_conv_2 * self.conv_output_size)
        #BatchNorm after FC and before Dropout
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.regularization_activation_function_fc1(x)
        #x = self.relu_fc1(x) 
        if self.dropout_probab>0: x = self.dropout1(x)
        #BatchNorm after FC and before Dropout
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.regularization_activation_function_fc2(x)
        #x = self.relu_fc2(x)
        if self.dropout_probab>0: x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

def weights_init_he(m):
    if isinstance(m, nn.Conv2d):
        #mode=fan_out: Used for convolutional layers to account for the output size of the layer
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            #nn.init.uniform_(m.bias, 0, 0.5)
            nn.init.constant_(m.bias, 0)
            #print(f"Convo Biases:\n{m.bias}")
    elif isinstance(m, nn.Linear):
        #mode=fan_in: Used for linear layers to account for the input size
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            #nn.init.uniform_(m.bias, 0, 0.5)
            nn.init.constant_(m.bias, 0)
            #print(f"Linear Biases:\n{m.bias}")

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

def Train_tail_end(epoch_cum_loss, epoch, best_cum_loss_epoch, best_cum_loss, train_loader, start_time, run_id, experiment_name,
                   net, stock_params):
    #end of training    
    print_mssg = f"End of Training: Cum Loss: {epoch_cum_loss} at {epoch}.<p>"
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

        mlflow.log_metric("epoch_cum_loss",epoch_cum_loss,step=epoch)
        mlflow.log_param(f"last_epoch", epoch)
        mlflow.log_param(f"best_final_cum_loss", best_cum_loss)
                                            
    helper_functions.save_checkpoint_model(best_cum_loss_epoch, best_cum_loss, epoch_cum_loss, net, run_id, experiment_name, stock_params)

def instantiate_optimizer_and_scheduler(net, params):
    if Parameters.run_adamw:
        Parameters.optimizer = optimizer = adamw.AdamW(net.parameters(), lr=params.learning_rate, weight_decay=Parameters.adamw_weight_decay)
        scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer, Parameters.batch_size, params.num_epochs_input, restart_period=Parameters.adamw_scheduler_restart_period, t_mult=Parameters.adamw_scheduler_t_mult, policy=Parameters.adamw_scheduler_cyclic_policy)
    else:
        if Parameters.optimizer_type == "Adam":
            Parameters.optimizer = optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)
        elif Parameters.optimizer_type == "SGD":
            Parameters.optimizer = optimizer = optim.SGD(net.parameters(), lr=params.learning_rate, momentum=params.momentum)
    
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=params.lr_scheduler_mode, patience=params.lr_scheduler_patience)
    
    return optimizer, scheduler

def Train(params, train_loader, net, run_id, experiment_name, device, stock_params):

    #enable grad
    torch.set_grad_enabled(True)

    inputs_list = []

    #best_cum_loss_epoch = 0
    best_cum_loss_epoch = torch.tensor(0.0, device=device, dtype=torch.float64)
    best_cum_loss = params.min_best_cum_loss
    best_checkpoint_cum_loss = params.best_checkpoint_cum_loss

    #torch.set_printoptions(threshold=torch.inf)

    start_time = time.time()

    print_mssg=f"Train params: learning_rate: {params.learning_rate}, momentum:{params.momentum} loss_threshold {params.loss_stop_threshold}<p>"
    #print(print_mssg)
    if params.save_runs_to_md:
        helper_functions.write_to_md(print_mssg,None)

    net.apply(weights_init_he)

    #criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    criterion = Parameters.function_loss

    # if Parameters.run_adamw:
    #     Parameters.optimizer = optimizer = adamw.AdamW(net.parameters(), lr=params.learning_rate, weight_decay=Parameters.adamw_weight_decay)
    #     scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer, Parameters.batch_size, params.num_epochs_input, restart_period=Parameters.adamw_scheduler_restart_period, t_mult=Parameters.adamw_scheduler_t_mult, policy=Parameters.adamw_scheduler_cyclic_policy)
    # else:
    #     if Parameters.optimizer_type == "Adam":
    #         Parameters.optimizer = optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)
    #     elif Parameters.optimizer_type == "SGD":
    #         Parameters.optimizer = optimizer = optim.SGD(net.parameters(), lr=params.learning_rate, momentum=params.momentum)
    
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=params.lr_scheduler_mode, patience=params.lr_scheduler_patience)
    
    optimizer, scheduler = instantiate_optimizer_and_scheduler(net, params)

    # profiler = torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
    # profiler.start()
    
    for epoch in range(params.num_epochs_input):

        if Parameters.run_adamw:
            scheduler.step()

        epoch_cum_loss = torch.tensor(0.0, device=device, dtype=torch.float64)

        gradients_dict = {name: [] for name, _ in net.named_parameters() if 'weight' in name}

        weights_dict = {name: [] for name, _ in net.named_parameters() if 'weight' in name}

        first_batch = True
        
        for i, data in enumerate(train_loader, 0):
            #print(f"Batch {i + 1}")

            #non_blocking combined with dataloader pin_memory for simultaneous transfer&computation
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            #print(f"Inputs shape: {inputs.shape}")
            #print(f"Labels shape: {labels.shape}")

            if Parameters.nn_predict_price==False:
                labels = labels.flatten().long()

            #mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.float16):

                if epoch == 0:
                    input_tensor = inputs.data
                    inputs_list.append(input_tensor)
                    
                # get signature for model
                if i==0 and epoch==0:
                    #print("epoch",epoch,"data i",i,"len image",len(inputs), "shape",inputs.shape, inputs)
                    #print("epoch",epoch,"data i",i,"label",labels,"labels shape",labels.shape, labels)
                    input_np = inputs.detach().cpu().numpy()
                    if Parameters.enable_mlflow:
                        model_signature = mlflow.models.infer_signature(input_np,net(inputs).detach().cpu().numpy())

                if Parameters.save_arch_bool:
                    helper_functions.Save_Model_Arch(net, run_id, inputs.shape, [inputs.dtype], "train", experiment_name)
                    Parameters.save_arch_bool = False
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                #print("outputs shape",outputs.shape,outputs)
                #print("outputs",outputs,"labels",labels)
                loss = criterion(outputs, labels)
            
            if loss is not None:
                loss.backward()
                optimizer.step()

                # Print optimizer's state_dict
                # print("Optimizer's state_dict:")
                # for var_name in optimizer.state_dict():
                #     print(var_name, "\t", optimizer.state_dict()[var_name])

                # print epoch/loss
                epoch_cum_loss += loss.detach()
                
                if first_batch and ((epoch + 1) % params.epoch_running_gradients_check == 0):
                    for name, param in net.named_parameters():
                        if 'weight' in name:
                            if param.grad is not None:
                                gradients_dict[name].append(param.grad.detach().clone().cpu().numpy())
                            weights_dict[name].append(param.detach().clone().cpu().numpy())

                    plot_data.plot_weights_gradients(weights_dict, gradients_dict, epoch,experiment_name, run_id)

                first_batch = False

            # torch.cuda.empty_cache()
            # _ = gc.collect() #force garbage collect

        # store inputs for metrics calculations e.g. correl
        if epoch == 0:
            stack_input = torch.stack(inputs_list, dim=0)

        # report curr cum loss
        if ((epoch + 1) % params.epoch_running_loss_check == 0):
                print_mssg = f"[{(epoch + 1):d}, {(i + 1):5d}] Cum loss: {(epoch_cum_loss):.9f}<p>"
                print(print_mssg) 
                if Parameters.save_runs_to_md:
                    helper_functions.write_to_md(print_mssg,None)

        # best_cum_loss improved
        if epoch_cum_loss < best_cum_loss:
            best_cum_loss_epoch = epoch
            best_cum_loss = epoch_cum_loss
            if Parameters.enable_mlflow:
                mlflow.log_metric("best_cum_loss",best_cum_loss,step=epoch)
        
        print(f"Cum loss: {epoch_cum_loss} epoch {epoch} Best Cum Loss: {best_cum_loss} at epoch {best_cum_loss_epoch}")
        
        if not Parameters.run_adamw:
            print("scheduler LR",scheduler.get_last_lr()[0])#"optimizer loss",optimizer.param_groups[0]['lr']

        # loss<threshold: update checkpoint
        if epoch_cum_loss < best_checkpoint_cum_loss:
            best_checkpoint_cum_loss = epoch_cum_loss
            best_cum_loss_epoch = epoch
            best_cum_loss = epoch_cum_loss
            helper_functions.update_best_checkpoint_dict(best_cum_loss_epoch, run_id, net.state_dict(), Parameters.optimizer.state_dict(), epoch_cum_loss)

        if epoch != 0 and epoch % Parameters.save_model_at_epoch_multiple == 0:
            helper_functions.save_checkpoint_model(best_cum_loss_epoch, best_cum_loss, epoch_cum_loss, net, run_id, experiment_name, stock_params)
        
        #exit if below loss threshold
        #print(f"Comparing Stop - Cum Loss {epoch_cum_loss} Vs {params.loss_stop_threshold}")
        if epoch_cum_loss < params.loss_stop_threshold:
            print(f"[WARNING] Epoch Cum Loss is less than {params.loss_stop_threshold}:{epoch_cum_loss} at {epoch}. Stopping training.")
            if Parameters.save_runs_to_md:
                helper_functions.write_to_md(f"Epoch Cum Loss is less than {params.loss_stop_threshold}:{epoch_cum_loss} at {epoch}. Stopping training.<p>",None)
            
            #profiler.stop()

            Train_tail_end(epoch_cum_loss, epoch, best_cum_loss_epoch, best_cum_loss, train_loader, start_time, run_id, experiment_name, net, stock_params)

            #return net, model_signature, stack_input
            return net, model_signature if 'model_signature' in locals() else None or None, stack_input
        
        #exit if there's no improvement for x-epochs and LR Reset not enabled
        if not params.enable_lr_reset and ((best_cum_loss_epoch + params.max_stale_loss_epochs) < epoch):
            print(f"[WARNING] Epoch Cum Loss Stale {epoch_cum_loss} at {epoch}. Abandon training.")
            if Parameters.save_runs_to_md:
                helper_functions.write_to_md(f"[WARNING] Epoch Cum Loss Stale {epoch_cum_loss} at {epoch}. Abandon training.",None)
            
            #profiler.stop()
            
            Train_tail_end(epoch_cum_loss, epoch, best_cum_loss_epoch, best_cum_loss, train_loader, start_time, run_id, experiment_name, net, stock_params)

            return net, model_signature if 'model_signature' in locals() else None, stack_input
        
        if Parameters.run_adamw:
            scheduler_param_group= scheduler.batch_step()
            print("AdamW - Getting Learning Rate ", scheduler_param_group['lr'])
            print("AdamW - Getting weight decay", scheduler_param_group['weight_decay'])
            epoch_metrics = {f"epoch_cum_loss": epoch_cum_loss.item(),
                        f"last_lr": scheduler_param_group['lr']}
        else:
            #optim learning rate scheduler for ReduceLROnPlateau
            scheduler.step(epoch_cum_loss.item())
            current_lr = scheduler.get_last_lr()[0]
            epoch_metrics = {f"epoch_cum_loss": epoch_cum_loss.item(),
                        f"last_lr": current_lr}
            
        if Parameters.enable_mlflow:
            mlflow.log_metrics(epoch_metrics,step=epoch)

        #LR reset
        if not Parameters.run_adamw:
            if params.enable_lr_reset and ((best_cum_loss_epoch + params.max_stale_loss_epochs) < epoch):
                optimizer.param_groups[0]['lr'] = 0.01#params.lr_reset_rate
                #scheduler.cooldown = params.lr_scheduler_patience
                print(f"[WARNING] Current Scheduler LR {scheduler.get_last_lr()[0]} reset as Cum Loss Stale {epoch_cum_loss} at {epoch}. LR reset to {optimizer.param_groups[0]['lr']}.")
            
            if params.enable_lr_reset and optimizer.param_groups[0]['lr'] <=params.lr_reset_threshold:
                print(f"Setting LR {optimizer.param_groups[0]['lr']} to Reset Rate {params.lr_reset_rate}")
                optimizer.param_groups[0]['lr']=params.lr_reset_rate

    #end training without reaching loss_threshold
    #profiler.stop()

    Train_tail_end(epoch_cum_loss, epoch, best_cum_loss_epoch, best_cum_loss, train_loader, start_time, run_id, experiment_name, net, stock_params)
    
    torch.set_printoptions()

    return net, model_signature if 'model_signature' in locals() else None, stack_input

def Test(test_loader, net, stock_ticker, device, experiment_name, run):
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

    torch.set_printoptions(threshold=torch.inf)

    for i, data in enumerate(test_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            outputs = net(inputs)
            #print("outputs",outputs)
        
        input_tensor = inputs.data
        actual_tensor = labels.data
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
            
            correct_1dp_tensor = (torch.abs(predicted_tensor - actual_tensor)<= 0.1)
            correct_1dp_list.append(correct_1dp_tensor)
        else:
            #print("predicted",predicted_tensor)
            #print("Actual",actual_tensor)
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
    print(); print(f"Stock {stock_ticker} Accuracy 2 decimal places: {correct_2dp_score}%, "
            f"Accuracy 1 decimal places: {correct_1dp_score}%, "
            f"Accuracy Classification {correct_classification_score}\n")

    text_mssg=(f"Stock {stock_ticker} Accuracy 2 decimal places: {correct_2dp_score}% <p> \
               Accuracy 1 decimal places: {correct_1dp_score}% <p> \
               Accuracy Classification {correct_classification_score} <p>")

    if Parameters.save_runs_to_md:
        helper_functions.write_to_md(text_mssg,None)

    # log metrics
    if Parameters.enable_mlflow:
        if Parameters.nn_predict_price:
            accuracy_metrics = {f"accuracy_2dp_score": correct_2dp_score.double().item(),
                    f"accuracy_1dp_score": correct_1dp_score.double().item(),
                    f"error_pct_outside_iqr": error_pct_outside_iqr}
        else:
            accuracy_metrics = {f"accuracy_classification_score": correct_classification_score.double().item()}
        
        mlflow.log_metrics(accuracy_metrics)
    
    # accuracyby_element_metrics = {}
    # pred_element_value = {}
    # for idx, sc in enumerate(torch.stack(correct_2dp_list, dim=0).flatten()):
    #     accuracyby_element_metrics[f'{stock_ticker}_accuracy_2dp_score_ele_{idx}'] = sc.double().item()
    #     accuracyby_element_metrics[f'{stock_ticker}_accuracy_1dp_score_ele_{idx}'] = sc.double().item()
    # for idx, pred in enumerate(torch.stack(predicted_list, dim=0).flatten()):
    #     pred_element_value[f'{stock_ticker}_pred_value_ele_{idx}'] = pred.double().item()
    # mlflow.log_metrics(accuracyby_element_metrics)
    # mlflow.log_metrics(pred_element_value)

    #print("abs_percentage_diffs",abs_percentage_diffs_np)

    return stack_input, predicted_list, actual_list, accuracy, stack_actual, stack_predicted

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