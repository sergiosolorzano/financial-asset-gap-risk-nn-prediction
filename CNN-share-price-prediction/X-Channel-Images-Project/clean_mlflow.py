#!/usr/bin/env python

import os
import shutil
import mlflow

def delete_experiment_artifacts(mlruns_directory, experiment_id):
    experiment_dir = os.path.join(mlruns_directory, '1', experiment_id)
    if os.path.exists(experiment_dir):
        print(f"Deleting artifacts for experiment {experiment_id}: {experiment_dir}")
        shutil.rmtree(experiment_dir)
    else:
        print(f"No artifacts found for experiment {experiment_id}")

# Example usage
mlruns_directory = 'mlruns'  # Path to your MLflow artifacts directory
experiment_id = '1'  # Replace with your experiment ID
#delete_experiment_artifacts(mlruns_directory, experiment_id)
experiments = mlflow.search_experiments(view_type=3)
print("exp",experiments)