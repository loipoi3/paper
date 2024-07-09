import argparse
from code.draw_graph import results_getter
from code.experiments.binary_classification_image_data.main import binary_image_pipeline
from code.experiments.binary_classification_tabular_data.main import binary_tabular_pipeline
from code.experiments.multiclass_classification_image_data.main import multiclass_image_pipeline
from code.experiments.multiclass_classification_tabular_data.main import multiclass_tabular_pipeline


def main():
    """
    Main function to parse arguments and execute the appropriate mode (train or display).
    """
    parser = argparse.ArgumentParser(description='Train models or display results.')
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')

    # Subparser for training models
    train_parser = subparsers.add_parser('train', help='Train the models')
    train_parser.add_argument('--task',
                              choices=['binary_tabular', 'binary_image', 'multiclass_tabular', 'multiclass_image'],
                              required=True, help='Task for which to train the models')
    train_parser.add_argument('--models', nargs='+', choices=['mlp_with_gd', 'mlp_with_spm', 'ea'], required=True,
                              help='Models to be trained (choices: mlp_with_gd - mlp with gradient descent, mlp_with_spm - mlp with single-point mutation, ea - (1+lambda)-evolutionary algorithm with genetic programming encoding)')
    train_parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for cross-validation')
    train_parser.add_argument('--mlp_with_gd_hidden_layers', type=int, nargs='+',
                              help='Hidden layers sizes for mlp_with_gd')
    train_parser.add_argument('--mlp_with_gd_activation', type=str, help='Activation function for mlp_with_gd')
    train_parser.add_argument('--mlp_with_gd_solver', type=str, default='sgd', help='Solver for mlp_with_gd')
    train_parser.add_argument('--mlp_with_gd_alpha', type=float, help='Alpha (regularization term) for mlp_with_gd')
    train_parser.add_argument('--mlp_with_gd_learning_rate_init', type=float,
                              help='Initial learning rate for mlp_with_gd')
    train_parser.add_argument('--mlp_with_gd_learning_rate', type=str, help='Learning rate schedule for mlp_with_gd')
    train_parser.add_argument('--mlp_with_gd_batch_size', type=int, help='Batch size for mlp_with_gd')
    train_parser.add_argument('--mlp_with_gd_tol', type=float, help='Tolerance for optimization for mlp_with_gd')
    train_parser.add_argument('--mlp_with_gd_n_iterations', type=int,
                              help='Maximum number of iterations for mlp_with_gd')
    train_parser.add_argument('--mlp_with_spm_hidden_layers', type=int, nargs='+',
                              help='Hidden layers sizes for mlp_with_spm')
    train_parser.add_argument('--mlp_with_spm_max_iter', type=int, help='Maximum number of iterations for mlp_with_spm')
    train_parser.add_argument('--mlp_with_spm_scale_for_mutation', type=float,
                              help='Scale for mutation for mlp_with_spm')
    train_parser.add_argument('--ea_tree_depth', type=int, help='Tree depth for ea')
    train_parser.add_argument('--ea_primitive_set', nargs='+', help='Primitive set for ea')
    train_parser.add_argument('--ea_terminal_set', nargs='+', help='Terminal set for ea')
    train_parser.add_argument('--ea_lambda', type=int, help='Lambda for ea')
    train_parser.add_argument('--ea_max_generations', type=int, help='Max generations for ea')
    train_parser.add_argument('--ea_save_checkpoint_path', type=str, help='Save checkpoint path for ea')

    # Subparser for displaying results
    display_parser = subparsers.add_parser('display', help='Display results')
    display_parser.add_argument('--path', type=str, required=True, help='Path to the results file')
    display_parser.add_argument('--iter_interval', type=int, required=True,
                                help='Iteration interval for displaying results')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'display':
        display(args)
    else:
        parser.print_help()


def train(args):
    """
    Trains the selected models based on the provided arguments.

    Args:
        args: Parsed command-line arguments.
    """
    print(f"Training task: {args.task}")

    if 'mlp_with_gd' in args.models:
        print("Training mlp_with_gd with the following hyperparameters:")
        print(f"Hidden layers: {args.mlp_with_gd_hidden_layers}")
        print(f"Activation function: {args.mlp_with_gd_activation}")
        print(f"Solver: {args.mlp_with_gd_solver}")
        print(f"Alpha: {args.mlp_with_gd_alpha}")
        print(f"Learning rate init: {args.mlp_with_gd_learning_rate_init}")
        print(f"Learning rate: {args.mlp_with_gd_learning_rate}")
        print(f"Max iterations: {args.mlp_with_gd_n_iterations}")
        print(f"Batch size: {args.mlp_with_gd_batch_size}")
        print(f"Tolerance: {args.mlp_with_gd_tol}\n")

    if 'mlp_with_spm' in args.models:
        print("Training mlp_with_spm with the following hyperparameters:")
        print(f"Hidden layers: {args.mlp_with_spm_hidden_layers}")
        print(f"Max iterations: {args.mlp_with_spm_max_iter}")
        print(f"Scale for mutation: {args.mlp_with_spm_scale_for_mutation}\n")

    if 'ea' in args.models:
        print("Training ea with the following hyperparameters:")
        print(f"Tree depth: {args.ea_tree_depth}")
        print(f"Primitive set: {args.ea_primitive_set}")
        print(f"Terminal set: {args.ea_terminal_set}")
        print(f"Lambda: {args.ea_lambda}")
        print(f"Max generations: {args.ea_max_generations}")
        print(f"Save checkpoint path: {args.ea_save_checkpoint_path}")

    # Execute the appropriate pipeline based on the task
    if args.task == 'binary_tabular':
        binary_tabular_pipeline(args)
    elif args.task == 'binary_image':
        binary_image_pipeline(args)
    elif args.task == 'multiclass_tabular':
        multiclass_tabular_pipeline(args)
    elif args.task == 'multiclass_image':
        multiclass_image_pipeline(args)


def display(args):
    """
    Displays the results based on the provided arguments.

    Args:
        args: Parsed command-line arguments.
    """
    results_getter(args.path, args.iter_interval)


if __name__ == '__main__':
    main()
