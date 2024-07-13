import json
import os
import pickle
import numpy as np
# from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from code.models.one_plus_lambda_ea_with_gp_encodings import GeneticAlgorithmModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from code.utils import logger

def run_one_plus_lambda_ea_with_gp(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                                   args) -> None:
    """
    Runs a (1 + lambda) Evolutionary Algorithm with Genetic Programming for binary classification of image data.

    Args:
        x_train: Training features.
        x_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        args: Arguments for configuring the EA and cross-validation.
    """
    print("(1 + lambda) - EA with GP for the problem of binary classification of image data:")

    # Configuration
    n_splits = args.n_splits
    max_generations = args.ea_max_generations if args.ea_max_generations is not None else 200
    tree_depth = args.ea_tree_depth if args.ea_tree_depth is not None else 6
    primitive_set = list(args.ea_primitive_set) if args.ea_primitive_set is not None else ["sub", "mul", "min", "max",
                                                                                           "hypot", "_safe_atan2",
                                                                                           "_float_lt", "_float_gt",
                                                                                           "_float_ge", "_float_le"]
    terminal_set = list(args.ea_terminal_set) if args.ea_terminal_set is not None else ["Constant_0", "E"]
    lambd = args.ea_lambda if args.ea_lambda is not None else 4
    save_checkpoint_path = args.ea_save_checkpoint_path if args.ea_save_checkpoint_path is not None else ""

    start = time.time()
    # Combine and split data for cross-validation
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # Initialize metrics
    overall_train_log_losses = np.zeros(max_generations)
    overall_test_log_losses = np.zeros(max_generations)
    total_time_list = np.zeros(max_generations)
    overall_accuracy, overall_precision, overall_recall, overall_f1_score = 0, 0, 0, 0

    # skf = StratifiedKFold(n_splits=n_splits)
    # splits = list(skf.split(X_pca, y))
    # with open(file_path, 'wb') as f:
    #     pickle.dump(splits, f)
    # print(f"Splits saved to {file_path}")

    # Load precomputed cross-validation splits
    file_path = "./experiments/binary_classification_image_data/checkpoints/cv/splits.pkl"
    with open(file_path, 'rb') as f:
        splits = pickle.load(f)

    for train_index, test_index in tqdm(splits, desc="Cross-validation folds"):
        # if fold_idx == 0 or fold_idx == 1 or fold_idx == 2:
        #    continue
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize Genetic Algorithm Model
        model = GeneticAlgorithmModel(x_train, y_train, x_test, y_test, tree_depth=tree_depth,
                                      primitive_set=primitive_set, terminal_set=terminal_set)

        champion, train_losses, test_losses, time_list = model.run(lambd=lambd, max_generations=max_generations,
                                                                   save_checkpoint_path=save_checkpoint_path)

        # if fold_idx == 3:
        # champion, train_losses, test_losses, time_list, fold_f1s, fold_accuracies, fold_precisions, fold_recalls = model.run(
        #     lambd=4, max_generations=n_iterations,
        #     save_checkpoint_path=f"/home/loipoi/bachelor-diploma/code/experiments/binary_classification_image_data/checkpoints/cv/{fold_idx}",
        #     save_checkpoint=True,
        #     start_checkpoint=f"/home/loipoi/bachelor-diploma/code/experiments/binary_classification_image_data/checkpoints/cv/{fold_idx}/checkpoint_gen_3999.pkl"
        # )
        # else:
        #    champion, train_losses, test_losses, time_list, fold_f1s, fold_accuracies, fold_precisions, fold_recalls = model.run(
        #         lambd=4, max_generations=n_iterations,
        #         save_checkpoint_path=f"/home/loipoi/bachelor-diploma/code/experiments/binary_classification_image_data/checkpoints/cv/{fold_idx}",
        #         save_checkpoint=True,
        #         start_checkpoint=f"/home/loipoi/bachelor-diploma/code/experiments/binary_classification_image_data/checkpoints/cv/{fold_idx}/checkpoint_gen_2999.pkl"
        #    )

        overall_train_log_losses += np.array(train_losses)
        overall_test_log_losses += np.array(test_losses)
        total_time_list += np.array(time_list)

        threshold = 0.0
        accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []

        while threshold <= 1.0:
            y_pred = model.make_predictions_with_threshold(champion, x_test, threshold=threshold)
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
    end = time.time()
    print(
        f"Accuracy={overall_accuracy:.4f}, Precision={overall_precision:.4f}, Recall={overall_recall:.4f}, F1-score={overall_f1_score:.4f}, Time={end-start}")

    results = {'train_losses': avg_train_log_losses.tolist(), 'test_losses': avg_test_log_losses.tolist(),
               'times': avg_time_list.tolist()}

    destination = "./results/binary_classification_image_data"
    if not os.path.exists(destination):
        os.makedirs(destination)

    results_file = os.path.join(destination, 'one_plus_lambda_ea_with_gp.json')
    with open(results_file, mode='w') as file:
        json.dump(results, file, indent=4)

    print(
        "Results (train loss list, test loss list, time list) are saved in the file: code/results/binary_classification_image_data/one_plus_lambda_ea_with_gp.json")
