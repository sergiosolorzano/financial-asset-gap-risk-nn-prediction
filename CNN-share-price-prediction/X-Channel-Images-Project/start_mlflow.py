#!/usr/bin/env python

import json
import subprocess
import os
import mlflow
import logging
import sys

import importlib as importlib

#print("***curre dir",os.getcwd())
sys.path.append(os.path.abspath('../../..'))
sys.path.append(os.path.abspath('./helper_functions_dir'))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..","..","..")))

#print("##",sys.path)
from helper_functions_dir import credentials

#logging.basicConfig(level=logging.DEBUG)
logging.getLogger("mlflow").setLevel(logging.DEBUG)

#current_dir = 
#print("current_dir",current_dir)
#print("creds_dir",creds_dir)
#creds_dir = os.path.abspath(os.path.join(os.getcwd(), "..","..",".."))
_credentials = credentials.MLflow_Credentials()
_credentials.get_credentials()

# with open(os.path.join(creds_dir,Parameters.mlflow_credentials_fname), 'r', encoding='utf-8') as file:
#     data = json.load(file)

#os.environ["AZURE_STORAGE_CONNECTION_STRING"] = f"set AZURE_STORAGE_CONNECTION_STRING={_credentials.AZURE_STORAGE_CONNECTION_STRING}"
os.environ["AZURE_STORAGE_ACCESS_KEY"] = f"{_credentials.AZURE_STORAGE_ACCESS_KEY}"
command = _credentials.command

if command:
    print("executing",command)
    
    process = subprocess.Popen(command, shell=True)
    process.wait()
else:
    print("No command found in the JSON file.")
