import matplotlib.pyplot as plt


def plot_losses(train_losses: list, test_losses: list):
    """
    Plots the training and test loss over iterations.

    Args:
        train_losses (list): List of training losses.
        test_losses (list): List of test losses.
    """
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 14})
    generations = range(1, len(train_losses) + 1)
    plt.plot(generations, train_losses, label='Training Loss')
    plt.plot(generations, test_losses, label='Test Loss')
    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Training and Test Loss over Iterations', fontsize=18)
    plt.legend(fontsize=14)
    # plt.xticks(list(range(1, 2)) + list(range(1000, len(train_losses) + 1, 1000)))
    plt.show()


def summarize_best_loss_performance(train_losses: list, test_losses: list, time_list: list, iter_interval: int):
    """
    Summarizes the best loss performance and prints iteration details at specified intervals.

    Args:
        train_losses (list): List of training losses.
        test_losses (list): List of test losses.
        time_list (list): List of times per iteration.
        iter_interval (int): Interval for displaying iteration details.
    """
    best_test_loss = min(test_losses)
    best_test_indexes = [i + 1 for i, loss in enumerate(test_losses) if loss == best_test_loss]
    total_times_up_to_best_test = [sum(time_list[:i]) for i in best_test_indexes]

    for i in range(1, len(train_losses) + 1):
        if i % iter_interval == 0:
            print("Iteration: ", i)
            print("Time: ", sum(time_list[:i]))
            print("Train Loss: ", train_losses[i - 1])
            print("Test Loss: ", test_losses[i - 1])

    print("Best Test Loss:", best_test_loss)
    print("Corresponding Train Loss:", train_losses[best_test_indexes[0] - 1])
    print("Indexes of Best Test Loss:", best_test_indexes)
    print("Total Times up to these iterations (seconds):", total_times_up_to_best_test)
    return best_test_loss
