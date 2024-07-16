import json
import os
import time
from typing import Any
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm


def run_classic_mlp(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                    args: Any) -> None:
    """
    Train a classic MLP model for multiclass classification on tabular data.

    Args:
        x_train: Training features.
        x_test: Testing features.
        y_train: Training labels.
        y_test: Testing labels.
        args: Command-line arguments with model hyperparameters.
    """
    print("Classic MLP for the problem of multiclass classification of tabular data:")

    # Extract hyperparameters from args
    n_iterations = args.mlp_with_gd_n_iterations if args.mlp_with_gd_n_iterations is not None else 734
    hidden_layer_sizes = tuple(args.mlp_with_gd_hidden_layers) if args.mlp_with_gd_hidden_layers is not None else (
        10, 10)
    activation = args.mlp_with_gd_activation if args.mlp_with_gd_activation is not None else "logistic"
    solver = args.mlp_with_gd_solver if args.mlp_with_gd_solver is not None else "sgd"
    alpha = args.mlp_with_gd_alpha if args.mlp_with_gd_alpha is not None else 0.00012710876052649163
    learning_rate_init = args.mlp_with_gd_learning_rate_init if args.mlp_with_gd_learning_rate_init is not None else 0.0080252458565383933
    learning_rate = args.mlp_with_gd_learning_rate if args.mlp_with_gd_learning_rate is not None else "adaptive"
    batch_size = args.mlp_with_gd_batch_size if args.mlp_with_gd_batch_size is not None else 64
    tol = args.mlp_with_gd_tol if args.mlp_with_gd_tol is not None else 0.0002348954721998801

    # Initialize the MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                        learning_rate_init=learning_rate_init, learning_rate=learning_rate, batch_size=batch_size,
                        tol=tol, max_iter=1, early_stopping=False, warm_start=True)

    train_log_losses, test_log_losses = [], []
    time_list = []

    for _ in tqdm(range(n_iterations), desc="Training iterations"):
        start_time = time.time()
        mlp.fit(x_train, y_train)
        time_list.append(time.time() - start_time)

        train_probs = mlp.predict_proba(x_train)
        test_probs = mlp.predict_proba(x_test)

        train_loss = log_loss(y_train, train_probs)
        test_loss = log_loss(y_test, test_probs)

        train_log_losses.append(train_loss)
        test_log_losses.append(test_loss)

    y_pred_test = mlp.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')

    print(
        f"Accuracy: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1-score: {f1_test:.4f}")

    results = {'train_losses': train_log_losses, 'test_losses': test_log_losses, 'times': time_list}

    destination = "./results/multiclass_classification_tabular_data"
    if not os.path.exists(destination):
        os.makedirs(destination)

    results_file = os.path.join(destination, 'classic_mlp.json')
    with open(results_file, mode='w') as file:
        json.dump(results, file, indent=4)

    print(
        "Results (train loss list, test loss list, time list) are saved in the file: mycode/results/multiclass_classification_tabular_data/classic_mlp.json")
