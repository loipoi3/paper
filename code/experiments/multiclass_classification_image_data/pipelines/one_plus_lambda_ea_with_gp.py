import json
import os
from typing import Any
from code.models.one_plus_lambda_ea_with_gp_encodings import GeneticAlgorithmModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def run_one_plus_lambda_ea_with_gp(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                                   args: Any) -> None:
    """
    Train a (1 + lambda) EA with GP model for multiclass classification on image data.

    Args:
        x_train: Training features.
        x_test: Testing features.
        y_train: Training labels.
        y_test: Testing labels.
        args: Command-line arguments with model hyperparameters.
    """
    print("(1 + lambda) - EA with GP for the problem of multiclass classification of image data:")

    # Extract hyperparameters from args
    max_generations = args.ea_max_generations if args.ea_max_generations is not None else 3
    tree_depth = args.ea_tree_depth if args.ea_tree_depth is not None else 8
    primitive_set = list(args.ea_primitive_set) if args.ea_primitive_set is not None else ["add", "sub", "mul",
                                                                                           "_safe_div", "min", "max",
                                                                                           "hypot", "logaddexp"]
    terminal_set = list(args.ea_terminal_set) if args.ea_terminal_set is not None else ["Constant_0", "Constant_1",
                                                                                        "Constant_minus_1"]
    lambd = args.ea_lambda if args.ea_lambda is not None else 4
    save_checkpoint_path = args.ea_save_checkpoint_path if args.ea_save_checkpoint_path is not None else ""

    # Initialize the Genetic Algorithm model
    model = GeneticAlgorithmModel(x_train, y_train, x_test, y_test, tree_depth=tree_depth,
                                  primitive_set=primitive_set, terminal_set=terminal_set,
                                  num_classes=len(np.unique(y_train)))
    champion, train_losses, test_losses, time_list = model.run(lambd=lambd, max_generations=max_generations,
                                                               save_checkpoint_path=save_checkpoint_path)

    y_pred_test = model.make_predictions_with_threshold(champion, x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')

    print(
        f"Accuracy: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1-score: {f1_test:.4f}")

    results = {'train_losses': train_losses, 'test_losses': test_losses, 'times': time_list}

    destination = "./results/multiclass_classification_image_data"
    if not os.path.exists(destination):
        os.makedirs(destination)

    results_file = os.path.join(destination, 'one_plus_lambda_ea_with_gp.json')
    with open(results_file, mode='w') as file:
        json.dump(results, file, indent=4)

    print(
        "Results (train loss list, test loss list, time list) are saved in the file: code/results/multiclass_classification_image_data/one_plus_lambda_ea_with_gp.json")
