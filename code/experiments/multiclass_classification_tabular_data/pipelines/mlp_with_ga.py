import json
import os
from typing import Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from code.models.mlp import MLP
import numpy as np


def run_mlp_with_ga(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                    args: Any) -> None:
    """
    Train an MLP model using Genetic Algorithm for multiclass classification on tabular data.

    Args:
        x_train: Training features.
        x_test: Testing features.
        y_train: Training labels.
        y_test: Testing labels.
        args: Command-line arguments with model hyperparameters.
    """
    print("MLP with GA for the problem of multiclass classification of tabular data:")

    # Extract hyperparameters from args
    n_iterations = args.mlp_with_spm_max_iter if args.mlp_with_spm_max_iter is not None else 40000
    hidden_layers = tuple(args.mlp_with_spm_hidden_layers) if args.mlp_with_spm_hidden_layers is not None else (
        x_train.shape[1], len(np.unique(y_train)))
    scale_for_mutation = args.mlp_with_spm_scale_for_mutation if args.mlp_with_spm_scale_for_mutation is not None else 0.1

    # Initialize the MLP model
    model = MLP(hidden_layer_sizes=hidden_layers, max_iter=n_iterations)
    model.fit(x_train, y_train, scale_for_mutation=scale_for_mutation, check_test_statistic=True, x_test=x_test,
              y_test=y_test)

    y_pred_test = model.predict(x_test, num_classes=len(np.unique(y_train)))
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')

    print(
        f"Accuracy: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1-score: {f1_test:.4f}")

    results = {'train_losses': model.errors, 'test_losses': model.errors_test, 'times': model.times}

    destination = "./results/multiclass_classification_tabular_data"
    if not os.path.exists(destination):
        os.makedirs(destination)

    results_file = os.path.join(destination, 'mlp_with_ga.json')
    with open(results_file, mode='w') as file:
        json.dump(results, file, indent=4)

    print(
        "Results (train loss list, test loss list, time list) are saved in the file: code/results/multiclass_classification_tabular_data/mlp_with_ga.json")
