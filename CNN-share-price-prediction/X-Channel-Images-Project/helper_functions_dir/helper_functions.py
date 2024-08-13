from __future__ import print_function

import os
import sys
import glob
import re
import json
import numpy as np

import random as rand
import matplotlib.pyplot as plt

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import pandas as pd
import io

import mlflow
from torchinfo import summary

#import scripts
import importlib as importlib
#sys.path.append(os.path.abspath('./helper_functions_dir'))
import plot_data as plot_data
#import parameters as params
from parameters import Parameters

import credentials

import torch
print(torch.__version__)

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

def Save_Model_Arch(net, run_id, input_shape, input_type, mode):
    summ = str(summary(net, input_size = input_shape, dtypes=input_type, mode=mode))
    model_arch_fname = f"./{Parameters.model_arch_dir}/model_arch_runid_{run_id}.txt"
    save_txt_to_blob(summ, os.path.basename(model_arch_fname))
    with open(model_arch_fname, "w", encoding="utf-8") as f:
        f.write(summ)
    mlflow.log_artifact(model_arch_fname)

def save_checkpoint_model(epoch, run_id, net, net_state_dict, opti_state_dict, epoch_loss):
    PATH = f'./{Parameters.checkpoint_dir}/model_checkpoint_runid_{run_id}_epoch_{epoch}.pth'
    torch.save({
            'run_id': run_id,
            'epoch': epoch,
            'model_state_dict': net_state_dict,
            'optimizer_state_dict': opti_state_dict,
            'loss': epoch_loss,
            }, PATH)
    mlflow.log_artifact(PATH, run_id=run_id)
    save_file_to_blob(PATH,os.path.basename(PATH))

def save_full_model(run_id, net):
    PATH = f'./{Parameters.full_model_dir}/model_full_runid_{run_id}.pth'
    torch.save(net, PATH)
    mlflow.pytorch.log_model(net, os.path.basename(PATH))

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
    with open(f'scenarios_output.txt', 'a') as file:
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

def write_scenario_to_log_file(accuracy):
    #write to file
    output_string = (f"Accuracy 2dp: {accuracy[0]}%\n"
                    f"Accuracy 1dp: {accuracy[1]}%\n")

    Scenario_Log(output_string)

def Save_Model(run_id, net):
    save_full_model(run_id,net)

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

    name, ext = os.path.splitext(f"{dir}/image.png")
    for filename in os.listdir(dir):
        number_str = filename[6:12]
        number = int(number_str)
        if number > latest_number:
            latest_number = number
    
    formatted_next_number = f"{latest_number+1:06d}"
    image_path = f'{name}_{formatted_next_number}{ext}'

    return image_path

def save_df_to_blob(raw_data, blob_name):
    csv_buffer = io.StringIO()
    raw_data.to_csv(csv_buffer, index=True)
    csv_data = csv_buffer.getvalue()

    connect_str = _credentials.AZURE_STORAGE_CONNECTION_STRING
    container_name = _credentials.storage_container_name
    directory = _credentials.blob_directory

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(f"{container_name}/{directory}")
    try:
        container_client.create_container()
    except Exception as e:
        print(f"Container already exists.")

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(csv_data, blob_type="BlockBlob", overwrite=True)

    print(f"File uploaded to Azure Blob Storage as {blob_name}")
    
    return blob_client.url

def save_txt_to_blob(text, blob_name):
    text_data = text.encode('utf-8')

    connect_str = _credentials.AZURE_STORAGE_CONNECTION_STRING
    container_name = _credentials.storage_container_name
    directory = _credentials.blob_directory

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(f"{container_name}/{directory}")
    try:
        container_client.create_container()
    except Exception as e:
        print(f"Container already exists.")

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(text_data, blob_type="BlockBlob", overwrite=True)

    print(f"File uploaded to Azure Blob Storage as {blob_name}")
    
    return blob_client.url

def save_file_to_blob(file_path, blob_name):
    with open(file_path, "rb") as file:
        file_data = file.read()
    
    connect_str = _credentials.AZURE_STORAGE_CONNECTION_STRING
    container_name = _credentials.storage_container_name
    directory = _credentials.blob_directory

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(f"{container_name}/{directory}")
    
    # Create the container if it does not exist
    try:
        container_client.create_container()
    except Exception as e:
        print(f"Container already exists or an error occurred: {e}")

    # Upload the file content to the blob
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(file_data, blob_type="BlockBlob", overwrite=True)

    print(f"File uploaded to Azure Blob Storage as {blob_name}")

def read_csv_from_blob(blob_name):
    connect_str = _credentials.AZURE_STORAGE_CONNECTION_STRING
    container_name = _credentials.storage_container_name
    directory = _credentials.blob_directory

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    #container_client = blob_service_client.get_container_client(container_name)
    
    full_blob_path = f"{directory}/{blob_name}" if directory else blob_name

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=full_blob_path)
    
    download_stream = blob_client.download_blob()
    csv_data = download_stream.readall().decode('utf-8')
    
    df = pd.read_csv(io.StringIO(csv_data))
    print("*****",df)

    return df

#inputinput_or_pred_price_or_image = input/prediction and state if price or image dataset
def mlflow_log_dataset(data_df, source, stock_ticker, input_or_pred_price_or_image, train_or_test_or_all, run, tags):
    
    data_df.info()
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