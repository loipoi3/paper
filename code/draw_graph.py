import json
from code.utils import plot_losses, summarize_best_loss_performance


def results_getter(path: str, iter_interval: int):
    """
    Loads results from a file and displays them.

    Args:
        path (str): Path to the results file.
        iter_interval (int): Interval for displaying iteration results.
    """
    with open(path, 'r') as file:
        data = json.load(file)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    times = data['times']
    plot_losses(train_losses, test_losses)
    summarize_best_loss_performance(train_losses, test_losses, times, iter_interval)
