from __future__ import print_function

import os
import sys
import time

import torch
print(torch.__version__)
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn

import numpy as np

import mlflow

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import plot_data as plot_data
import image_transform as image_transform
import helper_functions as helper_functions
import compute_stats as compute_stats
from parameters import Parameters

#set gpu env
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("At training - Device",device)
print("cuda version",torch.version.cuda)

class Net(nn.Module):
    def __init__(self, params):
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
        #filtersize,stride.
        #maxpool acts the same way in each channel, so doesn't need to be fed the num channels of the input
        self.pool = nn.MaxPool2d(kernel_size=params.filter_size_2, stride = params.stride_2)
        self.conv2 = nn.Conv2d(params.output_conv_1, params.output_conv_2, params.filter_size_3,params.stride_1)
        self.bn2 = nn.BatchNorm2d(params.output_conv_2)

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
        self.fc2 = nn.Linear(params.output_FC_1, params.output_FC_2)
        self.bn_fc2 = nn.BatchNorm1d(params.output_FC_2)
        self.fc3 = nn.Linear(params.output_FC_2, params.final_FCLayer_outputs)
        
        self.dropout1 = nn.Dropout(params.dropout_probab)
        self.dropout2 = nn.Dropout(params.dropout_probab)

        # compute the total number of parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #print(self.name + ': total params:', total_params)
        self.totalparams=total_params

    def forward(self, x):
        #BatchNorm after Conv and before Pooling
        x = F.relu(self.bn1(self.conv1(x)))  
        x = self.pool(x)
        #BatchNorm after Conv and before Pooling
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, self.output_conv_2 * self.conv_output_size)
        #BatchNorm after FC and before Dropout
        x = F.relu(self.bn_fc1(self.fc1(x)))
        if self.dropout_probab>0: x = self.dropout1(x)
        #BatchNorm after FC and before Dropout
        x = F.relu(self.bn_fc2(self.fc2(x)))
        if self.dropout_probab>0: x = self.dropout2(x)
        x = self.fc3(x)
        return x

def weights_init_he(m):
    if isinstance(m, nn.Conv2d):
        #mode=fan_out: Used for convolutional layers to account for the output size of the layer
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        #mode=fan_in: Used for linear layers to account for the input size
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def print_layer_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            print(f"{name}: {param.numel()} weights")

def instantiate_net(params):
    
    net = Net(params)
    
    net.to(device)
    net.parameters()
    #print_layer_weights(net)

    return net

def Train_tail_end(epoch_cum_loss, epoch, best_cum_loss_epoch, best_cum_loss, train_loader, start_time, run_id, experiment_name):
    #end of training    
    print_mssg = f"End of Training: Cum Loss: {epoch_cum_loss} at {epoch}.<p>"
    print(print_mssg)
    if Parameters.save_runs_to_md:
        helper_functions.write_to_md(print_mssg,None)

    mlflow.log_metric("epoch_cum_loss",epoch_cum_loss,step=epoch)
    mlflow.log_param(f"last_epoch", epoch)
    mlflow.log_param(f"best_final_cum_loss", best_cum_loss)

    helper_functions.save_checkpoint_model(best_cum_loss_epoch, run_id, experiment_name)
    
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")


def Train(params, train_loader, net, run_id, experiment_name):

    inputs_list = []

    best_cum_loss_epoch = 0
    best_cum_loss = params.min_best_cum_loss
    best_checkpoint_cum_loss = params.best_checkpoint_cum_loss

    #torch.set_printoptions(threshold=torch.inf)

    start_time = time.time()

    print_mssg=f"Train params: learning_rate: {params.learning_rate}, momentum:{params.momentum} loss_threshold {params.loss_threshold}<p>"
    #print(print_mssg)
    if params.save_runs_to_md:
        helper_functions.write_to_md(print_mssg,None)

    net.apply(weights_init_he)

    #criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    criterion = Parameters.function_loss

    if Parameters.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)
    elif Parameters.optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=params.learning_rate, momentum=params.momentum)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=params.lr_scheduler_mode, patience=params.lr_scheduler_partience)

    for epoch in range(params.num_epochs_input):

        epoch_cum_loss = 0.0

        gradients_dict = {name: [] for name, _ in net.named_parameters() if 'weight' in name}

        weights_dict = {name: [] for name, _ in net.named_parameters() if 'weight' in name}

        first_batch = True
        
        for i, data in enumerate(train_loader, 0):
            #print(f"Batch {i + 1}")
            
            inputs, labels = data[0].to(device), data[1].to(device)

            if epoch == 0:
                input_tensor = inputs.data
                input_tensor_cpu = input_tensor.cpu().detach()
                inputs_list.append(input_tensor_cpu)

            # get signature for model
            if i==0 and epoch==0:
                #print("epoch",epoch,"data i",i,"len image",len(inputs), "shape",inputs.shape, inputs)
                #print("epoch",epoch,"data i",i,"label",labels,"labels shape",labels.shape, labels)
                input_np = inputs.cpu().numpy()
                #pred_np = labels.cpu().numpy()
                model_signature = mlflow.models.infer_signature(input_np,net(inputs).detach().cpu().numpy())

            if Parameters.save_arch_bool:
                helper_functions.Save_Model_Arch(net, run_id, inputs.shape, [inputs.dtype], "train", experiment_name)
                Parameters.save_arch_bool = False
            
            # for e in data[1]:
            #     print("label",e.item())
            #print("label",data[1].item())
            #labels = labels.long()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            #print("outputs shape",outputs.shape,outputs)
            #print("outputs",outputs)
            loss = criterion(outputs, labels)
            if loss is not None:
                loss.backward()
                optimizer.step()

                # Print optimizer's state_dict
                # print("Optimizer's state_dict:")
                # for var_name in optimizer.state_dict():
                #     print(var_name, "\t", optimizer.state_dict()[var_name])

                # print epoch/loss
                epoch_cum_loss += loss.item()
                
                if first_batch and ((epoch + 1) % params.epoch_running_gradients_check == 0):
                    for name, param in net.named_parameters():
                        if 'weight' in name:
                            if param.grad is not None:
                                gradients_dict[name].append(param.grad.detach().cpu().numpy())
                            weights_dict[name].append(param.detach().cpu().numpy())

                    plot_data.plot_weights_gradients(weights_dict, gradients_dict, epoch,experiment_name, run_id)
    
                first_batch = False

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
            mlflow.log_metric("best_cum_loss",best_cum_loss,step=epoch)
        
        # loss<threshold: update checkpoint
        if epoch_cum_loss < best_checkpoint_cum_loss:
            best_checkpoint_cum_loss = epoch_cum_loss
            #best_cum_loss_epoch = epoch
            #best_cum_loss = epoch_cum_loss
            helper_functions.update_best_checkpoint_dict(best_cum_loss_epoch, run_id, net.state_dict(), optimizer.state_dict(), epoch_cum_loss)
        
        #exit if below loss threshold
        if (epoch_cum_loss) < params.loss_threshold:
            print(f"[WARNING] Epoch Cum Loss is less than {params.loss_threshold}:{epoch_cum_loss} at {epoch}. Stopping training.")
            if Parameters.save_runs_to_md:
                helper_functions.write_to_md(f"Epoch Cum Loss is less than {params.loss_threshold}:{epoch_cum_loss} at {epoch}. Stopping training.<p>",None)
            
            Train_tail_end(epoch_cum_loss, epoch, best_cum_loss_epoch, best_cum_loss, train_loader, start_time, run_id, experiment_name)

            return net, model_signature, stack_input
        
        #exit if there's no improvement for x-epochs
        if (best_cum_loss_epoch + params.max_stale_loss_epochs) < epoch:
            print(f"[WARNING] Epoch Cum Loss Stale {epoch_cum_loss} at {epoch}. Abandon training.")
            if Parameters.save_runs_to_md:
                helper_functions.write_to_md(f"[WARNING] Epoch Cum Loss Stale {epoch_cum_loss} at {epoch}. Abandon training.",None)
            
            Train_tail_end(epoch_cum_loss, epoch, best_cum_loss_epoch, best_cum_loss, train_loader, start_time, run_id, experiment_name)
            return net, model_signature, stack_input
        
        #optim learning rate scheduler
        scheduler.step(epoch_cum_loss)
        current_lr = scheduler.get_last_lr()[0]
        epoch_metrics = {f"epoch_cum_loss": epoch_cum_loss,
                         f"last_lr": current_lr}
        mlflow.log_metrics(epoch_metrics,step=epoch)

    #end training without reaching loss_threshold
    Train_tail_end(epoch_cum_loss, epoch, best_cum_loss_epoch, best_cum_loss, train_loader, start_time, run_id, experiment_name)

    #torch.set_printoptions()
                
    return net, model_signature, stack_input

def Test(test_loader, net, stock_ticker):
    inputs_list = []
    accuracy = 0
    correct_2dp_list = []
    correct_1dp_list = []
    error_list = []
    #dataframe vars
    predicted_list = []
    actual_list = []
    abs_percentage_diffs_list = []
    stack_input=None
    stack_predicted=None

    torch.set_printoptions(threshold=torch.inf)

    for i, data in enumerate(test_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            outputs = net(inputs)
        
        input_tensor = inputs.data
        predicted_tensor = outputs.data
        actual_tensor = labels.data
        
        input_tensor_cpu = input_tensor.cpu().detach()
        inputs_list.append(input_tensor_cpu)

        predicted_tensor_cpu = predicted_tensor.cpu().detach()
        predicted_list.append(predicted_tensor_cpu)

        actual_tensor_cpu = actual_tensor.cpu().detach()
        actual_list.append(actual_tensor_cpu)
        
        #calculate batch pct difference
        batch_abs_percentage_diff = (torch.abs(predicted_tensor - actual_tensor) / torch.abs(actual_tensor)) * 100
        abs_percentage_diffs_list.extend(batch_abs_percentage_diff.cpu().detach().numpy().flatten())

        #Check IQR
        error_tensor = (predicted_tensor - actual_tensor).cpu().detach()
        error_list.append(error_tensor)
        
        #accuracy
        correct_2dp_tensor = (torch.abs(predicted_tensor - actual_tensor)<= 0.01).cpu().detach()
        correct_2dp_list.append(correct_2dp_tensor)
        
        correct_1dp_tensor = (torch.abs(predicted_tensor - actual_tensor)<= 0.1).cpu().detach()
        correct_1dp_list.append(correct_1dp_tensor)

    torch.set_printoptions()

    correct_2dp_score = calculate_score(correct_2dp_list)
    correct_1dp_score = calculate_score(correct_1dp_list)

    mean_correct_2dp_score = torch.mean(correct_2dp_score, dim=0)
    mean_correct_1dp_score = torch.mean(correct_1dp_score, dim=0)

    #print("PREDICTED",predicted_list,"shape",predicted_list[0].shape)
    mean_predicted = [tensor.mean() for tensor in predicted_list]
    
    error_list_iqr, error_pct_outside_iqr = compute_stats.calculate_iqr(error_list)

    text_mssg=f"error_pct_outside_iqr {error_pct_outside_iqr}<p>"
    if Parameters.save_runs_to_md:
        helper_functions.write_to_md(text_mssg,None)

    stack_input = torch.stack(inputs_list, dim=0)
    
    #actual and predicted mean
    # print("actual list",actual_list)
    # print("predicted list",predicted_list)
    stack_actual = torch.stack(actual_list, dim=0)
    #print("stack actual shape",stack_actual.shape,"stack actual",stack_actual)

    stack_predicted = torch.stack(predicted_list, dim=0)
    #print("stack predicted shape",stack_predicted.shape)#,"stack_predicted",stack_predicted)
    #print("stack input shape",stack_input.shape)#,"stack_input",stack_input)
        
    accuracy = [correct_2dp_score, correct_1dp_score]
    print(); print(f"Mean accuracy 2 decimal places: {mean_correct_2dp_score}%, "
            f"Mean accuracy 1 decimal places: {mean_correct_1dp_score}%,\n",
            f"Percentage of predictions within ",
            f"2 decimal places: {correct_2dp_score}%, "
            f"1 decimal places: {correct_1dp_score}%,\n")

    text_mssg=(f"Mean accuracy 2 decimal places: {mean_correct_2dp_score}% <p> \
               Mean accuracy 1 decimal places: {mean_correct_1dp_score}% <p> \
                Percentage of predictions within 2 decimal places: {correct_2dp_score}% <p> \
                    1 decimal places: {correct_1dp_score}%<p>")
    if Parameters.save_runs_to_md:
        helper_functions.write_to_md(text_mssg,None)

    # log metrics
    accuracy_metrics = {f"{stock_ticker}_mean_accuracy_2dp_score": mean_correct_2dp_score.double().item(),
               f"{stock_ticker}_mean_accuracy_1dp_score": mean_correct_1dp_score.double().item(),
               f"{stock_ticker}_error_pct_outside_iqr": error_pct_outside_iqr}
    mlflow.log_metrics(accuracy_metrics)
    
    accuracyby_element_metrics = {}
    pred_element_value = {}
    for idx, sc in enumerate(correct_2dp_score):
        accuracyby_element_metrics[f'{stock_ticker}_accuracy_2dp_score_ele_{idx}'] = correct_2dp_score[idx].double().item()
        accuracyby_element_metrics[f'{stock_ticker}_accuracy_1dp_score_ele_{idx}'] = correct_1dp_score[idx].double().item()
    for idx, val in enumerate(mean_predicted):
        pred_element_value[f'{stock_ticker}_pred_value_ele_{idx}'] = mean_predicted[idx].double().item()
    mlflow.log_metrics(accuracyby_element_metrics)
    mlflow.log_metrics(pred_element_value)

    #print("abs_percentage_diffs",abs_percentage_diffs_np)
    return stack_input, predicted_list, actual_list, accuracy, stack_actual, stack_predicted

def calculate_score(tensor_list):
    #print("tensor_list",tensor_list)
    stack = torch.stack(tensor_list, dim=1)
    #print("len stack",len(stack.data[0]))
    #print("true_sum",torch.sum(stack,dim=1))
    true_sum = torch.sum(stack,dim=1)
    score = (true_sum/len(stack.data[0]))*100
    return score

def set_model_for_eval(net):
    #set model for evaluation
    net.eval()
    
    return net