import os
import json
import sys

import importlib as importlib
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../../../'))
from parameters import Parameters

class MLflow_Credentials():
    def __init__(self):
        self.command = None
        self.AZURE_STORAGE_CONNECTION_STRING = None
        self.username_db = None
        self.password_db = None
        self.server_db = None
        self.sqljdb_connector_location = None
        self.mlflow_db = None
        self.port_db = None
        self.blob_directory = None
        self.storage_container_name = None

    def get_credentials(self):
        current_dir = os.getcwd()
        #print("Curre dir",current_dir)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        #print("creds path",script_dir)
        creds_dir = os.path.abspath(os.path.join(current_dir, "..","..",".."))
        #print("creds dir",creds_dir)
        with open(os.path.join(creds_dir,Parameters.mlflow_credentials_fname), 'r', encoding='utf-8') as file:
            data = json.load(file)

        self.command = data.get('command')
        #print("got command",self.command)
        self.AZURE_STORAGE_CONNECTION_STRING = data.get('AZURE_STORAGE_CONNECTION_STRING')
        self.username_db = data.get('username_db')
        self.password_db = data.get('password_db')
        self.server_db = data.get('server_db')
        self.sqljdb_connector_location = data.get('sqljdb_connector_location')
        self.mlflow_db = data.get('mlflow_db')
        self.port_db = data.get('port_db')
        self.blob_directory = data.get('blob_directory')
        self.storage_container_name = data.get('storage_container_name')