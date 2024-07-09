import json
import os
import time
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm


def run_classic_mlp(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, args) -> None:
    """
    Runs a classic MLP for binary classification of image data.

    Args:
        x_train: Training features.
        x_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        args: Arguments for configuring the MLP and cross-validation.
    """
    print("Classic MLP for the problem of binary classification of image data:")

    # Configuration
    n_splits = args.n_splits
    n_iterations = args.mlp_with_gd_n_iterations if args.mlp_with_gd_n_iterations is not None else 500
    hidden_layer_sizes = tuple(args.mlp_with_gd_hidden_layers) if args.mlp_with_gd_hidden_layers is not None else (
        15, 20, 15)
    activation = args.mlp_with_gd_activation if args.mlp_with_gd_activation is not None else "tanh"
    solver = args.mlp_with_gd_solver if args.mlp_with_gd_solver is not None else "sgd"
    alpha = args.mlp_with_gd_alpha if args.mlp_with_gd_alpha is not None else 0.005383724166734261
    learning_rate_init = args.mlp_with_gd_learning_rate_init if args.mlp_with_gd_learning_rate_init is not None else 0.0015898533701208645
    learning_rate = args.mlp_with_gd_learning_rate if args.mlp_with_gd_learning_rate is not None else "invscaling"
    batch_size = args.mlp_with_gd_batch_size if args.mlp_with_gd_batch_size is not None else 256
    tol = args.mlp_with_gd_tol if args.mlp_with_gd_tol is not None else 0.00032994812784605145

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

        # Initialize MLP
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                            learning_rate_init=learning_rate_init, learning_rate=learning_rate, batch_size=batch_size,
                            tol=tol, max_iter=1, early_stopping=False, warm_start=True)

        fold_train_losses, fold_test_losses = [], []
        time_list = []

        for _ in tqdm(range(n_iterations), desc="Training iterations"):
            start_time = time.time()
            mlp.fit(x_train, y_train)
            iteration_time = time.time() - start_time
            time_list.append(iteration_time)

            train_probs = mlp.predict_proba(x_train)
            test_probs = mlp.predict_proba(x_test)

            train_loss = log_loss(y_train, train_probs)
            test_loss = log_loss(y_test, test_probs)

            fold_train_losses.append(train_loss)
            fold_test_losses.append(test_loss)

        overall_train_log_losses += np.array(fold_train_losses)
        overall_test_log_losses += np.array(fold_test_losses)
        total_time_list += np.array(time_list)

        threshold = 0.0
        accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []

        while threshold <= 1.0:
            y_prob = mlp.predict_proba(x_test)[:, 1]
            y_pred = (y_prob > threshold).astype(int)

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

    results_file = os.path.join(destination, 'classic_mlp.json')
    with open(results_file, mode='w') as file:
        json.dump(results, file, indent=4)

    print(
        "Results (train loss list, test loss list, time list) are saved in the file: code/results/binary_classification_image_data/classic_mlp.json")
