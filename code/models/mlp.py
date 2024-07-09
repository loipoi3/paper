import torch
import torch.nn.functional as f
import numpy as np
from tqdm import tqdm
import time


class MLP:
    """
    Multi-layer Perceptron (MLP) for binary and multi-class classification tasks.
    """

    def __init__(self, hidden_layer_sizes: tuple, max_iter: int):
        """
        Initializes the MLP with the given hidden layer sizes and maximum number of iterations.

        Args:
            hidden_layer_sizes (tuple): Tuple representing the size of each hidden layer.
            max_iter (int): Maximum number of training iterations.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self._weights = self._initialize_weights()
        self.errors = None
        self.errors_test = None
        self.times = None

    def _initialize_weights(self) -> list:
        """
        Initializes the weights of the MLP using Xavier initialization.

        Returns:
            list: List of weight tensors for each layer.
        """
        weights = []
        for idx in range(len(self.hidden_layer_sizes) - 1):
            # Calculate standard deviation for Xavier initialization
            xavier_stddev = np.sqrt(6.0 / (self.hidden_layer_sizes[idx] + self.hidden_layer_sizes[idx + 1]))
            # Initialize weights with uniform distribution
            weight = torch.tensor(np.random.uniform(-xavier_stddev, xavier_stddev,
                                                    (self.hidden_layer_sizes[idx], self.hidden_layer_sizes[idx + 1])),
                                  dtype=torch.float32)
            weights.append(weight)
        return weights

    def _eval_model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the MLP on the input data.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Output predictions of the MLP.
        """
        output = x
        for layer in range(len(self._weights) - 1):
            # Forward pass through hidden layers
            output = torch.mm(output, self._weights[layer])
            output = torch.sigmoid(output)
        # Forward pass through output layer
        output = torch.mm(output, self._weights[-1])
        if self._weights[-1].shape[1] == 1:
            output = torch.sigmoid(output)
        else:
            output = f.softmax(output, dim=1)
        return output

    def fit(self, x: np.ndarray, y: np.ndarray, scale_for_mutation: float, check_test_statistic: bool = False,
            x_test: np.ndarray = None, y_test: np.ndarray = None):
        """
        Trains the MLP using Single-Point Mutation (SPM).

        Args:
            x (np.ndarray): Training data.
            y (np.ndarray): Training labels.
            scale_for_mutation (float): Scale for mutation.
            check_test_statistic (bool, optional): Whether to check test statistics during training. Defaults to False.
            x_test (np.ndarray, optional): Test data. Defaults to None.
            y_test (np.ndarray, optional): Test labels. Defaults to None.
        """
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        if x_test is not None and y_test is not None:
            x_test = torch.tensor(x_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.long)

        self.errors = []
        self.times = []
        initial_loss = MLP._log_loss(y, self._eval_model(x))
        if check_test_statistic:
            self.errors_test = []

        for _ in tqdm(range(0, self.max_iter), desc="Training iterations"):
            start_time = time.time()
            # Select random layer, neuron and feature to mutate
            layer_index = np.random.randint(0, len(self._weights), size=1)[0]
            feature_index = np.random.randint(0, self._weights[layer_index].shape[0], size=1)[0]
            neuron_index = np.random.randint(0, self._weights[layer_index].shape[1], size=1)[0]
            value_of_mutation = np.random.normal(scale=scale_for_mutation)
            original_value = self._weights[layer_index][feature_index, neuron_index].item()
            # Apply mutation
            self._weights[layer_index][feature_index, neuron_index] += value_of_mutation
            current_loss = MLP._log_loss(y, self._eval_model(x))

            if current_loss < initial_loss:
                # Keep mutation if it improves the loss
                initial_loss = current_loss
            else:
                # Revert mutation if it does not improve the loss
                self._weights[layer_index][feature_index, neuron_index] = original_value

            self.errors.append(initial_loss.item())
            if check_test_statistic:
                current_test_loss = MLP._log_loss(y_test, self._eval_model(x_test))
                self.errors_test.append(current_test_loss.item())
            self.times.append(time.time() - start_time)

    def predict(self, x: np.ndarray, threshold: float = None, num_classes: int = 1) -> np.ndarray:
        """
        Predicts the output for the given input data.

        Args:
            x (np.ndarray): Input data.
            threshold (float, optional): Threshold for binary classification. Defaults to None.
            num_classes (int, optional): Number of output classes. Defaults to 1.

        Returns:
            np.ndarray: Predicted labels.
        """
        x = torch.tensor(x, dtype=torch.float32)
        output = self._eval_model(x).detach()
        if num_classes == 1:
            # Binary classification
            y_pred = (output >= threshold).int().numpy()
        else:
            # Multi-class classification
            y_pred = torch.argmax(output, axis=1).numpy()
        return y_pred

    @staticmethod
    def _log_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Computes the log loss between true and predicted labels.

        Args:
            y_true (torch.Tensor): True labels.
            y_pred (torch.Tensor): Predicted labels.

        Returns:
            torch.Tensor: Computed log loss.
        """
        if y_pred.shape[1] == 1:
            y_true = y_true.float().unsqueeze(1)
            loss_fn = torch.nn.BCELoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y_true)
        return loss
