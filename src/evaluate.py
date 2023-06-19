# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse

from azureml.core import Run

import joblib
import pandas as pd
from matplotlib import pyplot as plt
import mlflow

import aml_utils


DIR_FIGURES = 'figures/'


def main(model_path, dataset_path, output_dir):
    """Evaluate the model.

    Args:
        model_path (str): The path of the model file
        dataset_path (str): The path of the dataset to use for evaluation
        output_dir (str): The path of the output directory

    Returns:
        None

    """
    
    experiment_id = mlflow.create_experiment(
        "mlflow-experiment",
        artifact_location=DIR_FIGURES
    )
    experiment = mlflow.get_experiment(experiment_id)
    
    # debug
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("Creation timestamp: {}".format(experiment.creation_time))
    
    mlflow.set_experiment(experiment_id)
    #print("Experiment_id: {}".format(active_experiment.experiment_id))
    
    mlflow.start_run() # Start an MLflow run
    
    # Debug
    try:
        run = mlflow.active_run()
        run_id = run.info.run_id
        print(run_id)
        print(run)
    except:
        print("error encountered when trying to print active run")
    
    ws = aml_utils.retrieve_workspace()

    print("Loading model...")
    model = joblib.load(model_path)

    print("Reading test data...")
    data = pd.read_csv(dataset_path)

    print("Evaluating model...")
    y_test, X_test = split_data_features(data)
    metrics, plots = get_model_evaluation(model, X_test, y_test)
    print(metrics)

    # Save metrics MLFlow
    print("Saving metrics...")
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
     
    # Save figures in run outputs
    print(f"Saving figures in folder {DIR_FIGURES}...")
    os.makedirs(DIR_FIGURES, exist_ok=True)
    for fig_name, fig in plots.items():
        file_path = os.path.join(DIR_FIGURES, f'{fig_name}.png')
        fig.savefig(file_path)
        mlflow.log_artifact(file_path, artifact_path=DIR_FIGURES)
    
    mlflow.end_run()  # End the MLflow run
    print('Finished.')


def split_data_features(data):
    # Do your X/y features split here
    y_test, X_test = data.iloc[:, 0], data.iloc[:, 1:]
    return y_test, X_test


def get_model_evaluation(model, X_test, y_test):
    # Evaluate your model here
    metrics = { 'examplemetric1': 0.1, 'examplemetric2': 2.2 }
    plots = {
        'scatter': pd.DataFrame({'pred': [1, 0.2, 0.3], 'real': [0.9, 0.15, 0.5]}) \
                        .plot(x='real', y='pred', kind='scatter', figsize=(5, 5)) \
                        .get_figure()
    }
    return metrics, plots


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./outputs')
    args_parsed = parser.parse_args(args_list)
    return args_parsed


if __name__ == '__main__':
    args = parse_args()

    main(
        model_path=os.path.join(args.model_dir, f'{args.model_name}.pkl'),  # Path as defined in train.py
        dataset_path=os.path.join(args.dataset, 'dataset.csv'),  # Path as defined in dataprep.py
        output_dir=args.output_dir
    )
