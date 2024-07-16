import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from mycode.models.mlp import MLP


def run_mlp_with_ga(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, args) -> None:
    """
    Runs an MLP with Genetic Algorithm for binary classification of image data.

    Args:
        x_train: Training features.
        x_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        args: Arguments for configuring the MLP and cross-validation.
    """
    print("MLP with GA for the problem of binary classification of image data:")

    # Configuration
    n_splits = args.n_splits
    n_iterations = args.mlp_with_spm_max_iter if args.mlp_with_spm_max_iter is not None else 10000
    hidden_layers = tuple(args.mlp_with_spm_hidden_layers) if args.mlp_with_spm_hidden_layers is not None else (
        x_train.shape[1], 15, 20, 15, 1)
    scale_for_mutation = args.mlp_with_spm_scale_for_mutation if args.mlp_with_spm_scale_for_mutation is not None else 0.1

    # Combine and split data for cross-validation
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    skf = StratifiedKFold(n_splits=n_splits)

    # Initialize metrics
    overall_train_log_losses = np.zeros(n_iterations)
    overall_test_log_losses = np.zeros(n_iterations)
    total_time_list = np.zeros(n_iterations)
    overall_accuracy, overall_precision, overall_recall, overall_f1_score = 0, 0, 0, 0

    for train_index, test_index in tqdm(skf.split(x, y), desc="Cross-validation folds"):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize MLP with Genetic Algorithm
        model = MLP(hidden_layer_sizes=hidden_layers, max_iter=n_iterations)

        model.fit(x_train, y_train, check_test_statistic=True, x_test=x_test, y_test=y_test,
                  scale_for_mutation=scale_for_mutation)

        overall_train_log_losses += np.array(model.errors)
        overall_test_log_losses += np.array(model.errors_test)
        total_time_list += np.array(model.times)

        threshold = 0.0
        accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []

        while threshold <= 1.0:
            y_pred = model.predict(x_test, threshold=threshold)
            accuracy_lst.append((accuracy_score(y_test, y_pred), threshold))
            precision_lst.append((precision_score(y_test, y_pred, zero_division=0), threshold))
            recall_lst.append((recall_score(y_test, y_pred), threshold))
            f1_lst.append((f1_score(y_test, y_pred), threshold))

            threshold += 0.01

        max_f1_score = max(f1_lst, key=lambda i: i[0])[0]
        best_thresholds = [threshold for f1, threshold in f1_lst if f1 == max_f1_score]
        index = [i for i, (f1, threshold) in enumerate(f1_lst) if threshold == best_thresholds[0]][0]

        overall_accuracy += accuracy_lst[index][0]
        overall_precision += precision_lst[index][0]
        overall_recall += recall_lst[index][0]
        overall_f1_score += f1_lst[index][0]

    overall_accuracy = overall_accuracy / n_splits
    overall_precision = overall_precision / n_splits
    overall_recall = overall_recall / n_splits
    overall_f1_score = overall_f1_score / n_splits

    avg_train_log_losses = overall_train_log_losses / n_splits
    avg_test_log_losses = overall_test_log_losses / n_splits
    avg_time_list = total_time_list / n_splits

    print(
        f"Accuracy={overall_accuracy:.4f}, Precision={overall_precision:.4f}, Recall={overall_recall:.4f}, F1-score={overall_f1_score:.4f}")

    results = {'train_losses': avg_train_log_losses.tolist(), 'test_losses': avg_test_log_losses.tolist(),
               'times': avg_time_list.tolist()}

    destination = "./results/binary_classification_image_data"
    if not os.path.exists(destination):
        os.makedirs(destination)

    results_file = os.path.join(destination, 'mlp_with_ga.json')
    with open(results_file, mode='w') as file:
        json.dump(results, file, indent=4)

    print(
        "Results (train loss list, test loss list, time list) are saved in the file: mycode/results/binary_classification_image_data/mlp_with_ga.json")
