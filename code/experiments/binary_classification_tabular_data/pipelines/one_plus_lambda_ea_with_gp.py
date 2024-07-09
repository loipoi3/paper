import json
import os
import numpy as np
from tqdm import tqdm
from code.models.one_plus_lambda_ea_with_gp_encodings import GeneticAlgorithmModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold


def run_one_plus_lambda_ea_with_gp(x: np.ndarray, y: np.ndarray, args) -> None:
    """
    Runs a (1 + lambda) Evolutionary Algorithm with Genetic Programming for binary classification of tabular data.

    Args:
        x: Features.
        y: Labels.
        args: Arguments for configuring the EA and cross-validation.
    """
    print("(1 + lambda) - EA with GP for the problem of binary classification of tabular data:")

    # Configuration
    n_splits = args.n_splits
    max_generations = args.ea_max_generations if args.ea_max_generations is not None else 200
    tree_depth = args.ea_tree_depth if args.ea_tree_depth is not None else 3
    primitive_set = list(args.ea_primitive_set) if args.ea_primitive_set is not None else ["add", "mul", "min", "max",
                                                                                           "logaddexp", "_safe_atan2",
                                                                                           "_float_gt", "_float_ge",
                                                                                           "_safe_fmod"]
    terminal_set = list(args.ea_terminal_set) if args.ea_terminal_set is not None else ["Constant_1", "E"]
    lambd = args.ea_lambda if args.ea_lambda is not None else 2
    save_checkpoint_path = args.ea_save_checkpoint_path if args.ea_save_checkpoint_path is not None else ""

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_splits)

    # Initialize metrics
    overall_train_log_losses = np.zeros(max_generations)
    overall_test_log_losses = np.zeros(max_generations)
    total_time_list = np.zeros(max_generations)
    overall_accuracy, overall_precision, overall_recall, overall_f1_score = 0, 0, 0, 0

    for train_index, test_index in tqdm(skf.split(x, y), desc="Cross-validation folds"):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize Genetic Algorithm Model
        model = GeneticAlgorithmModel(x_train, y_train, x_test, y_test, tree_depth=tree_depth,
                                      primitive_set=primitive_set, terminal_set=terminal_set)

        champion, train_losses, test_losses, time_list = model.run(lambd=lambd, max_generations=max_generations,
                                                                   save_checkpoint_path=save_checkpoint_path)

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

    print(
        f"Accuracy={overall_accuracy:.4f}, Precision={overall_precision:.4f}, Recall={overall_recall:.4f}, F1-score={overall_f1_score:.4f}")

    results = {'train_losses': avg_train_log_losses.tolist(), 'test_losses': avg_test_log_losses.tolist(),
               'times': avg_time_list.tolist()}

    destination = "./results/binary_classification_tabular_data"
    if not os.path.exists(destination):
        os.makedirs(destination)

    results_file = os.path.join(destination, 'one_plus_lambda_ea_with_gp.json')
    with open(results_file, mode='w') as file:
        json.dump(results, file, indent=4)

    print(
        "Results (train loss list, test loss list, time list) are saved in the file: code/results/binary_classification_tabular_data/one_plus_lambda_ea_with_gp.json")
