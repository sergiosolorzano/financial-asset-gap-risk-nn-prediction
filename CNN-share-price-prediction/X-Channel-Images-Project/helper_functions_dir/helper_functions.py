from __future__ import print_function

import os
import sys
import glob
import re
import json
import mlflow.data.dataset_source_registry
import numpy as np
from pathlib import Path
import uuid

import time
import matplotlib
import matplotlib.pyplot as plt

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import pandas as pd
import io

import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from torchinfo import summary

#import scripts
import importlib as importlib
import plot_data as plot_data

from parameters import Parameters
import neural_network_enhanced as neural_network

matplotlib.use(Parameters.matplotlib_use)

import credentials

import torch
import random
import copy

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

_credentials = credentials.MLflow_Credentials()
_credentials.get_credentials()

def Delete_Scenario_Files():
    scenario_files = glob.glob('*scenario*') #find files
    iteration_files = glob.glob('*iteration*')
    optimizer_files = glob.glob('*optimizer*')
    model_files = glob.glob('*model*')

    files_to_delete = scenario_files + iteration_files + optimizer_files + model_files
    files_to_delete = [file for file in files_to_delete if file != 'model_card.md']
    
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")

def Save_BayesOpt_Model(scenario, net):
    PATH = f'./model_bayesOpt_iteration_{scenario}.pth'
    torch.save(net.state_dict(), PATH)

def Save_Model_Arch(net, run_id, input_shape, input_type, mode, experiment_name):
    summ = str(summary(net, input_size = input_shape, dtypes=input_type, mode=mode))

    os.makedirs(Parameters.model_arch_dir, exist_ok=True)
    model_arch_fname_with_dir = f"{Parameters.model_arch_dir}/{Parameters.model_arch_fname}.txt"
    
    #save_txt_to_blob(summ, os.path.basename(model_arch_fname), run_id, experiment_name)
    with open("./" + model_arch_fname_with_dir, "w", encoding="utf-8") as f:
        f.write(summ)
    
    if Parameters.enable_mlflow:
        mlflow.log_artifact(local_path="./" + model_arch_fname_with_dir, run_id=run_id, artifact_path=Parameters.model_arch_dir)


def update_best_checkpoint_dict(best_cum_loss_epoch, eval_max_r2_epoch, run_id, net_state_dict, opti_state_dict, epoch_loss):
    #print("Update checkpoint_dict epoch", best_cum_loss_epoch, "epoch loss", epoch_loss.item())
    #print("opti_state_dict",opti_state_dict)
    best_checkpt_dict = {
            'run_id': run_id,
            'epoch': best_cum_loss_epoch,
            'model_state_dict': copy.deepcopy(net_state_dict),
            'optimizer_state_dict': copy.deepcopy(opti_state_dict),
            'loss': epoch_loss,
            }
    print(f"Updated at best r^2 eval epoch {eval_max_r2_epoch} but best cum loss is at epoch {best_cum_loss_epoch}")# checkpt {best_checkpt_dict['model_state_dict']['conv2.weight'][0][0]}")
    
    return best_checkpt_dict

def save_checkpoint_model(best_checkpoint_dict, best_cum_loss_epoch, eval_max_r2_epoch, best_cum_loss, curr_epoch_cum_loss, net, run_id, experiment_name, stock_params, epoch):
    train_stocks = stock_params.train_stock_tickers
    eval_stocks = stock_params.eval_stock_tickers
    model_checkpoint_fname_with_dir = f'{Parameters.checkpoint_dir}/{Parameters.model_checkpoint_fname}_{train_stocks}_{eval_stocks}_{Parameters.model_uuid}.pth'
    # if Parameters.checkpt_dict['optimizer_state_dict'] == None:
    #     #print("***Updating checkpoint dict cos it's NONE", Parameters.checkpt_dict['optimizer_state_dict'])
    #     update_best_checkpoint_dict(best_cum_loss_epoch, run_id, net.state_dict(), Parameters.optimizer.state_dict(), curr_epoch_cum_loss)
    #print("SAVING:Parameters.checkpt_dict",Parameters.checkpt_dict['model_state_dict']['conv2.weight'])
    print("Saving model best eval R^2",eval_max_r2_epoch, " though best cum loss is", best_cum_loss.item(), "at Epoch ", best_cum_loss_epoch, "name",model_checkpoint_fname_with_dir)#, "best_checkpoint_dict",best_checkpoint_dict['model_state_dict']['conv2.weight'][0][0])
    torch.save(best_checkpoint_dict, "./" + model_checkpoint_fname_with_dir)
    
    if Parameters.enable_mlflow:
        #blob_with_dirs = "models" + "/" + f'{Parameters.model_checkpoint_fname}.pth'
        blob_with_dirs = Path("models", f'{Parameters.model_checkpoint_fname}.pth')
        if Parameters.enable_save_model:
            mlflow.log_artifact(local_path="./" + model_checkpoint_fname_with_dir, run_id=run_id, artifact_path=Parameters.checkpoint_dir)
        mlflow.set_tag(f"best_checkpoint_epoch", best_cum_loss_epoch)
        #save_file_to_blob(PATH,os.path.basename(PATH), run_id, experiment_name)

def save_full_model(run_id, net, model_signature, experiment_name, stock_params):
    train_stocks = stock_params.train_stock_tickers
    eval_stocks = stock_params.eval_stock_tickers
    PATH = f'./{Parameters.full_model_dir}/{Parameters.model_full_fname}_{train_stocks}_{eval_stocks}.pth'
    torch.save(net, PATH)
    pip_requirements = ['torch==2.3.0+cu121','torchvision==0.18.0+cu121']
    
    if Parameters.enable_mlflow:
        #blob_with_dirs = "models" + "/" + f'{Parameters.model_full_fname}.pth'
        mlflow.pytorch.log_model(
        pytorch_model=net,
        artifact_path=Parameters.full_model_dir,
        pip_requirements=pip_requirements,
        signature=model_signature)
        #mlflow.pytorch.log_model(net, blob_with_dirs,pip_requirements=pip_requirements, signature=model_signature)

def load_checkpoint_model(net, device, stock_params, train_loader):
    train_stocks = stock_params.train_stock_tickers
    eval_stocks = stock_params.eval_stock_tickers
    #instantiate optimizer used to train the model (ensure opt is correct in params)
    neural_network.instantiate_optimizer_and_scheduler(net, Parameters, train_loader)

    #load checkpoint
    model_checkpoint_fname_with_dir = f'{Parameters.checkpoint_dir}/{Parameters.model_checkpoint_fname}_{train_stocks}_{eval_stocks}_{Parameters.model_uuid}.pth'
    #print(f"Loading model {model_checkpoint_fname_with_dir}")
    checkpoint = torch.load(model_checkpoint_fname_with_dir, map_location=device)
    #load state dict and optimizer
    net.load_state_dict(checkpoint['model_state_dict'])
    
    #load weights
    Parameters.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded Model checkpt {checkpoint['model_state_dict']['conv2.weight'][0][0]}")

    #if further training required
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"\033[32mLoaded checkpoint from {model_checkpoint_fname_with_dir}, epoch: {epoch}, loss: {loss}\033[0m")
    
    return net, epoch, loss, checkpoint

def save_feature_maps(feature_maps_cnn_list, feature_maps_fc_list):
    cnn_arrays = [tensor.cpu().numpy() for tensor in feature_maps_cnn_list]
    np.savez(f"{Parameters.checkpoint_dir}/feature_maps_cnn_{Parameters.train_tickers}_{Parameters.eval_tickers}.npz", *cnn_arrays)

    # Save feature_maps_fc_list to a file
    fc_arrays = [tensor.cpu().numpy() for tensor in feature_maps_fc_list]
    np.savez(f"{Parameters.checkpoint_dir}/feature_maps_fc_{Parameters.train_tickers}_{Parameters.eval_tickers}.npz", *fc_arrays)

# def Load_State_Model(net, PATH):
#     print("Loading State Model")
#     net.load_state_dict(torch.load(PATH))
#     net.eval()
#     return net

# def Load_Full_Model(PATH):
#     if os.path.exists(PATH):
#         print("Loading Full Model")
#         net = torch.load(PATH)
#         net.eval()
#         return net
#     else:
#         print(f"Model {PATH} does not exist")

# def Clear_Scenario_Log(scenario):
#     with open(f'scenario_{scenario}.txt', 'w') as file:
#         file.write('')

def Scenario_Log(output_string):
    with open(Parameters.training_analytics_params_log_fname, 'a') as file:
        file.write('\n\n' + output_string)

def Load_BayesOpt_Model(scenario, net):
    PATH = f'./bayesian_optimization_saved_models/model_bayesOpt_iteration_{scenario}.pth'
    print("path model is ",PATH)
    net.load_state_dict(torch.load(PATH))
    net.eval()
    return net

def data_to_array(input_list):
    output_array = np.array(input_list)

    return output_array

def write_scenario_to_log_file(training_stats):
    #write to file
    # output_string = (f"Accuracy 2dp: {accuracy[0]}%\n"
    #                 f"Accuracy 1dp: {accuracy[1]}%\n",
    #                 f"Classification Accuracy: {accuracy[2]}%\n")

    Scenario_Log(str(training_stats))

def Save_Model(run_id, net, model_signature, experiment_name, stock_params):
    save_full_model(run_id,net, model_signature, experiment_name, stock_params)

def write_to_md(text, image_path):
    if text.strip():
        with open(Parameters.brute_force_filename, 'a') as f:
            f.write(text)
    if image_path != None:
        image_write = f"\n![Example Plot]({image_path})"
        with open(Parameters.brute_force_filename, 'a') as f:
            f.write(image_write)

def get_next_image_number():
    latest_number = 0
    dir = Parameters.brute_force_image_mlflow_dir
    os.makedirs(dir, exist_ok=True)

    name, ext = os.path.splitext(f"{dir}/image.png")
    for filename in os.listdir(dir):
        number_str = filename[6:12]
        number = int(number_str)
        if number > latest_number:
            latest_number = number
    
    formatted_next_number = f"{latest_number+1:06d}"
    image_path = f'{name}_{formatted_next_number}{ext}'

    return image_path

def write_and_log_plt(fig, epoch, name, md_name, experiment_name, run_id):
    #write image to md
    if Parameters.save_runs_to_md:
        image_path = get_next_image_number()
        plt.savefig(image_path, dpi=300)
        write_to_md(md_name,image_path)
    
    #if epoch is None:
        #mlflow.log_figure(fig, f"{Parameters.brute_force_image_mlflow_dir}/image_{name}.png")
    if Parameters.enable_mlflow:
        blob_with_dirs = f"images/image_{name}_{experiment_name}.png"
        #print("MLFLOW Image ",blob_with_dirs)
        mlflow.log_figure(fig, blob_with_dirs)
    # else:    
    #     #blob_with_dirs = _credentials.blob_directory + "/" + experiment_name + "/" + run_id + "/" + "images" + "/" + f"image_{name}_epoch_{epoch}.png"
    #     blob_with_dirs = f"images/image_{name}_epoch_{epoch}.png"
    #     mlflow.log_figure(fig, blob_with_dirs)
    #plt.show()
    plt.close(fig)

def save_df_to_blob(raw_data, blob_name, run_id, experiment_name):
    csv_buffer = io.StringIO()
    raw_data.to_csv(csv_buffer, index=True)
    csv_data = csv_buffer.getvalue()

    access_key = _credentials.AZURE_STORAGE_ACCESS_KEY
    container_name = _credentials.storage_container_name
    directory = _credentials.blob_directory
    storage_account = _credentials.storage_account_name

    container_url_with_sas = f"https://{storage_account}.blob.core.windows.net?{access_key}"
    blob_service_client = BlobServiceClient(account_url=container_url_with_sas)
    container_client = blob_service_client.get_container_client(container_name)
    
    blob_with_dirs = directory + "/" + run_id + "/artifacts/data/" + blob_name
    blob_client = container_client.get_blob_client(blob_with_dirs)
    try:
        blob_client.upload_blob(csv_data, blob_type="BlockBlob", overwrite=True)
        print("Upload succeeded.")
    except Exception as e:
        print(f"Error upload_blob during upload: {e}")

    print(f"File uploaded {blob_name}")
    
    return blob_client.url

def save_txt_to_blob(text, blob_name, run_id, experiment_name):
    text_data = text.encode('utf-8')

    # connect_str = _credentials.AZURE_STORAGE_CONNECTION_STRING
    # container_name = _credentials.storage_container_name
    # directory = _credentials.blob_directory

    # blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # container_client = blob_service_client.get_container_client(f"{container_name}/{directory}")
    # try:
    #     container_client.create_container()
    # except Exception as e:
    #     print(f"Container already exists.")

    # blob_client = container_client.get_blob_client(blob_name)
    # blob_client.upload_blob(text_data, blob_type="BlockBlob", overwrite=True)
    # blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    access_key = _credentials.AZURE_STORAGE_ACCESS_KEY
    container_name = _credentials.storage_container_name
    directory = _credentials.blob_directory
    storage_account = _credentials.storage_account_name

    container_url_with_sas = f"https://{storage_account}.blob.core.windows.net?{access_key}"
    blob_service_client = BlobServiceClient(account_url=container_url_with_sas)
    container_client = blob_service_client.get_container_client(container_name)

    blob_client = container_client.get_blob_client(directory + "/" + experiment_name + "/" + run_id +"/"+blob_name)
    blob_client.upload_blob(text_data, blob_type="BlockBlob", overwrite=True)

    print(f"File uploaded to Azure Blob Storage as {blob_name}")
    
    return blob_client.url

def save_file_to_blob(file_path, blob_name, run_id, experiment_name):
    with open(file_path, "rb") as file:
        file_data = file.read()
    
    # connect_str = _credentials.AZURE_STORAGE_CONNECTION_STRING
    # container_name = _credentials.storage_container_name
    # directory = _credentials.blob_directory

    # blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # container_client = blob_service_client.get_container_client(f"{container_name}/{directory}")
    
    # # Create the container if it does not exist
    # try:
    #     container_client.create_container()
    # except Exception as e:
    #     print(f"Container already exists")

    # # Upload the file content to the blob
    # blob_client = container_client.get_blob_client(blob_name)
    # blob_client.upload_blob(file_data, blob_type="BlockBlob", overwrite=True)

    # print(f"File uploaded to Azure Blob Storage as {blob_name}")
    access_key = _credentials.AZURE_STORAGE_ACCESS_KEY
    container_name = _credentials.storage_container_name
    directory = _credentials.blob_directory
    storage_account = _credentials.storage_account_name

    container_url_with_sas = f"https://{storage_account}.blob.core.windows.net?{access_key}"
    blob_service_client = BlobServiceClient(account_url=container_url_with_sas)
    container_client = blob_service_client.get_container_client(container_name)

    blob_client = container_client.get_blob_client(directory + "/" + experiment_name + "/" + run_id + "/" + blob_name)
    blob_client.upload_blob(file_data, blob_type="BlockBlob", overwrite=True)

    print(f"File uploaded to Azure Blob Storage as {blob_name}")

#inputinput_or_pred_price_or_image = input/prediction and state if price or image dataset
def mlflow_log_dataset(data_df, source, stock_ticker, input_or_pred_price_or_image, train_or_test_or_all, run, tags):
    
    #data_df.info()
    dataset = mlflow.data.from_pandas(
        data_df, source=source, name=f"{stock_ticker}_{input_or_pred_price_or_image}_{train_or_test_or_all}")
    mlflow.log_input(dataset, context=train_or_test_or_all)
    # logged_run = mlflow.get_run(run.info.run_id)

    # print("**",logged_run.inputs.dataset_inputs)
    # logged_dataset = logged_run.inputs.dataset_inputs[0].dataset
    # print("logged_dataset", logged_dataset)

    # # View some of the recorded Dataset information
    # print(f"Dataset name: {logged_dataset.name}")
    # print(f"Dataset digest: {logged_dataset.digest}")
    # print(f"Dataset profile: {logged_dataset.profile}")
    # print(f"Dataset schema: {logged_dataset.schema}")
    # print(f"dataset tags", logged_run.inputs.dataset_inputs[0].tags)