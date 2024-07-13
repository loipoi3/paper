import time
import numpy as np
import operator
import math
from deap import gp, creator, base, tools
from sklearn.metrics import log_loss
from scipy.special import softmax
import random
import pickle
import os
from tqdm import tqdm

from code.config import tracker


class GeneticAlgorithmModel:
    """
    Genetic Algorithm model for classification tasks using genetic programming.
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                 tree_depth: int, primitive_set: list, terminal_set: list, num_classes: int = 1):
        """
        Initializes the Genetic Algorithm model with training and test data, and other hyperparameters.

        Args:
            x_train (np.ndarray): Training data.
            y_train (np.ndarray): Training labels.
            x_test (np.ndarray): Test data.
            y_test (np.ndarray): Test labels.
            tree_depth (int): Depth of the trees in the genetic programming population.
            primitive_set (list): List of primitives to be used.
            terminal_set (list): List of terminals to be used.
            num_classes (int, optional): Number of output classes. Defaults to 1.
        """
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test
        self.tree_depth = tree_depth
        self._num_classes = num_classes
        self._primitive_set = primitive_set
        self._terminal_set = terminal_set

        self.pset = gp.PrimitiveSet("MAIN", x_train.shape[1])
        self._setup_primitives()

        if hasattr(creator, 'FitnessMin'):
            del creator.FitnessMin
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if hasattr(creator, 'Individual'):
            del creator.Individual
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self._setup_toolbox()

    def _setup_primitives(self):
        """
        Sets up the primitive set for the genetic programming algorithm.
        """
        if "add" in self._primitive_set:
            self.pset.addPrimitive(operator.add, 2)
        if "sub" in self._primitive_set:
            self.pset.addPrimitive(operator.sub, 2)
        if "mul" in self._primitive_set:
            self.pset.addPrimitive(operator.mul, 2)
        if "_safe_div" in self._primitive_set:
            self.pset.addPrimitive(GeneticAlgorithmModel._safe_div, 2)
        if "min" in self._primitive_set:
            self.pset.addPrimitive(min, 2)
        if "max" in self._primitive_set:
            self.pset.addPrimitive(max, 2)
        if "hypot" in self._primitive_set:
            self.pset.addPrimitive(math.hypot, 2)
        if "logaddexp" in self._primitive_set:
            self.pset.addPrimitive(np.logaddexp, 2)
        if "_safe_atan2" in self._primitive_set:
            self.pset.addPrimitive(GeneticAlgorithmModel._safe_atan2, 2)
        if "_float_lt" in self._primitive_set:
            self.pset.addPrimitive(GeneticAlgorithmModel._float_lt, 2)
        if "_float_gt" in self._primitive_set:
            self.pset.addPrimitive(GeneticAlgorithmModel._float_gt, 2)
        if "_float_ge" in self._primitive_set:
            self.pset.addPrimitive(GeneticAlgorithmModel._float_ge, 2)
        if "_float_le" in self._primitive_set:
            self.pset.addPrimitive(GeneticAlgorithmModel._float_le, 2)
        if "_safe_fmod" in self._primitive_set:
            self.pset.addPrimitive(GeneticAlgorithmModel._safe_fmod, 2)

        if "Constant_0" in self._terminal_set:
            self.pset.addTerminal(0, "Constant_0")
        if "Constant_1" in self._terminal_set:
            self.pset.addTerminal(1, "Constant_1")
        if "Constant_minus_1" in self._terminal_set:
            self.pset.addTerminal(-1, "Constant_minus_1")
        if "Pi" in self._terminal_set:
            self.pset.addTerminal(math.pi, "Pi")
        if "E" in self._terminal_set:
            self.pset.addTerminal(math.e, "E")

    def _setup_toolbox(self):
        """
        Sets up the toolbox for the genetic programming algorithm.
        """
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=self.tree_depth, max_=self.tree_depth)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=1)
        self.toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)

    @staticmethod
    @tracker.track_runtime
    def _safe_div(x: float, y: float) -> float:
        """
        Safely performs division, returning 1 if division by zero occurs.

        Args:
            x (float): Numerator.
            y (float): Denominator.

        Returns:
            float: Result of the division or 1 if division by zero occurs.
        """
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            result = np.divide(x, y)
            if np.isnan(result) or np.isinf(result):
                return 1
            return result

    @staticmethod
    @tracker.track_runtime
    def _safe_atan2(y: float, x: float) -> float:
        """
        Safely computes the arc tangent of y/x, returning 1 if both inputs are zero.

        Args:
            y (float): Y-coordinate.
            x (float): X-coordinate.

        Returns:
            float: Arc tangent of y/x or 1 if both inputs are zero.
        """
        if y == 0 and x == 0:
            return 1
        return math.atan2(y, x)

    @staticmethod
    @tracker.track_runtime
    def _float_lt(a: float, b: float) -> float:
        """
        Returns 1.0 if a < b, else 0.0.

        Args:
            a (float): First value.
            b (float): Second value.

        Returns:
            float: 1.0 if a < b, else 0.0.
        """
        return 1.0 if operator.lt(a, b) else 0.0

    @staticmethod
    @tracker.track_runtime
    def _float_gt(a: float, b: float) -> float:
        """
        Returns 1.0 if a > b, else 0.0.

        Args:
            a (float): First value.
            b (float): Second value.

        Returns:
            float: 1.0 if a > b, else 0.0.
        """
        return 1.0 if operator.gt(a, b) else 0.0

    @staticmethod
    @tracker.track_runtime
    def _float_ge(a: float, b: float) -> float:
        """
        Returns 1.0 if a >= b, else 0.0.

        Args:
            a (float): First value.
            b (float): Second value.

        Returns:
            float: 1.0 if a >= b, else 0.0.
        """
        return 1.0 if operator.ge(a, b) else 0.0

    @staticmethod
    @tracker.track_runtime
    def _float_le(a: float, b: float) -> float:
        """
        Returns 1.0 if a <= b, else 0.0.

        Args:
            a (float): First value.
            b (float): Second value.

        Returns:
            float: 1.0 if a <= b, else 0.0.
        """
        return 1.0 if operator.le(a, b) else 0.0

    @staticmethod
    @tracker.track_runtime
    def _safe_fmod(x: float, y: float) -> float:
        """
        Safely performs modulo operation, returning 1.0 if denominator is zero.

        Args:
            x (float): Numerator.
            y (float): Denominator.

        Returns:
            float: Result of the modulo operation or 1.0 if denominator is zero.
        """
        if y == 0:
            return 1.0
        return math.fmod(x, y)

    @staticmethod
    @tracker.track_runtime
    def _sigmoid(x: np.ndarray) -> float:
        """
        Computes the sigmoid function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            float: Sigmoid of the input value.
        """
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    @tracker.track_runtime
    def _evaluate_individual(self, individual, x: np.ndarray, y: np.ndarray) -> tuple:
        """
        Evaluates an individual in the population.

        Args:
            individual: Individual to evaluate.
            x (np.ndarray): Input data.
            y (np.ndarray): Labels.

        Returns:
            tuple: Log loss of the individual.
        """
        if self._num_classes == 1:
            func = self.toolbox.compile(expr=individual)
            # vectorized_func = np.vectorize(func)
            # predictions = self._sigmoid(vectorized_func(*np.hsplit(X, X.shape[1])))
            predictions = np.array([GeneticAlgorithmModel._sigmoid(func(*record)) for record in x])
        else:
            funcs = [self.toolbox.compile(expr=tree) for tree in individual["ind"]]
            predictions = np.array([[func(*record) for func in funcs] for record in x])
            predictions = softmax(predictions, axis=1)
        return log_loss(y, predictions),

    def run(self, lambd: int, max_generations: int, save_checkpoint_path: str, start_checkpoint: str = "",
            save_checkpoint: bool = False) -> tuple:
        """
        Runs the genetic algorithm.

        Args:
            lambd (int): Number of offsprings in each generation.
            max_generations (int): Maximum number of generations.
            save_checkpoint_path (str): Path to save checkpoints.
            start_checkpoint (str, optional): Path to start checkpoint. Defaults to "".
            save_checkpoint (bool, optional): Whether to save checkpoints. Defaults to False.

        Returns:
            tuple: Best individual, training losses, test losses, and time per generation.
        """
        if start_checkpoint != "":
            # Load from checkpoint if provided
            state = GeneticAlgorithmModel._load_checkpoint(start_checkpoint)
            champion = state['champion']
            start_generation = state['generation'] + 1
            train_losses = state['train_losses']
            test_losses = state['test_losses']
            time_list = state['time_list']
        else:
            train_losses, test_losses, time_list = [], [], []
            if self._num_classes == 1:
                champion = self.toolbox.individual()
                champion.fitness.values = self._evaluate_individual(champion, self.X_train, self.y_train)
            else:
                champion = {"ind": [self.toolbox.individual() for _ in range(self._num_classes)],
                            "fitness": {"values": None}}
                champion_fitness = self._evaluate_individual(champion, self.X_train, self.y_train)
                champion["fitness"]["values"] = champion_fitness[0]
            start_generation = 0

        for gen in tqdm(range(start_generation, max_generations), desc="Generations"):
            start_time = time.time()
            if self._num_classes == 1:
                candidates = [self.toolbox.clone(champion) for _ in range(1 + lambd)]
                # Mutate each candidate for binary classification
                for candidate in candidates:
                    self.toolbox.mutate(candidate)
                    del candidate.fitness.values
            else:
                # Mutate a random tree in each candidate for multi-class classification
                candidates = [
                    {"ind": [self.toolbox.clone(tree) for tree in champion["ind"]], "fitness": {"values": None}} for _
                    in range(1 + lambd)]
                for candidate in candidates:
                    selected_tree = random.choice(candidate["ind"])
                    self.toolbox.mutate(selected_tree)
                    del selected_tree.fitness.values

            # Evaluate fitness of each candidate
            if self._num_classes == 1:
                for candidate in candidates:
                    candidate.fitness.values = self._evaluate_individual(candidate, self.X_train, self.y_train)
            else:
                for candidate in candidates:
                    candidate["fitness"]["values"] = self._evaluate_individual(candidate, self.X_train, self.y_train)[0]

            candidates.append(champion)
            # Select the best candidate as the new champion
            if self._num_classes == 1:
                champion = tools.selBest(candidates, 1)[0]
            else:
                sorted_list = sorted(candidates, key=lambda x: x["fitness"]["values"])
                champion = sorted_list[0]

            time_list.append(time.time() - start_time)
            train_loss = self._evaluate_individual(champion, self.X_train, self.y_train)[0]
            train_losses.append(train_loss)
            test_loss = self._evaluate_individual(champion, self.X_test, self.y_test)[0]
            test_losses.append(test_loss)

            if save_checkpoint:
                GeneticAlgorithmModel._save_checkpoint(champion, gen, train_losses, test_losses, time_list,
                                                       save_checkpoint_path)

        # Print the average execution time for each tracked function
        print(f"Average execution time for _safe_div: {tracker.get_average_time('_safe_div'):.4f} seconds")
        print(f"Average execution time for _safe_atan2: {tracker.get_average_time('_safe_atan2'):.4f} seconds")
        print(f"Average execution time for _float_lt: {tracker.get_average_time('_float_lt'):.4f} seconds")
        print(f"Average execution time for _float_gt: {tracker.get_average_time('_float_gt'):.4f} seconds")
        print(f"Average execution time for _float_ge: {tracker.get_average_time('_float_ge'):.4f} seconds")
        print(f"Average execution time for _float_le: {tracker.get_average_time('_float_le'):.4f} seconds")
        print(f"Average execution time for _safe_fmod: {tracker.get_average_time('_safe_fmod'):.4f} seconds")
        print(f"Average execution time for _sigmoid: {tracker.get_average_time('_sigmoid'):.4f} seconds")
        print(
            f"Average execution time for _evaluate_individual: {tracker.get_average_time('_evaluate_individual'):.4f} seconds")

        # Reset the execution times
        tracker.execution_times = {}

        return champion, train_losses, test_losses, time_list

    def make_predictions_with_threshold(self, individual, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Makes predictions using the trained individual with a threshold for binary classification.

        Args:
            individual: Trained individual.
            x (np.ndarray): Input data.
            threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

        Returns:
            np.ndarray: Predicted labels.
        """
        if self._num_classes == 1:
            func = self.toolbox.compile(expr=individual)
            predictions_raw = np.array([func(*record) for record in x])
            predictions = GeneticAlgorithmModel._sigmoid(predictions_raw)
            return (predictions > threshold).astype(int)
        else:
            funcs = [self.toolbox.compile(expr=tree) for tree in individual["ind"]]
            predictions_raw = np.array([[func(*record) for func in funcs] for record in x])
            predictions = softmax(predictions_raw, axis=1)
            return np.argmax(predictions, axis=1)

    @staticmethod
    def _save_checkpoint(champion, generation: int, train_losses: list, test_losses: list, time_list: list,
                         directory: str):
        """
        Saves the current state of the genetic algorithm to a checkpoint file.

        Args:
            champion: Current best individual.
            generation (int): Current generation number.
            train_losses (list): List of training losses.
            test_losses (list): List of test losses.
            time_list (list): List of times per generation.
            directory (str): Directory to save the checkpoint.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, f"checkpoint_gen_{generation}.pkl")
        with open(filename, 'wb') as cp_file:
            pickle.dump({"champion": champion, "generation": generation, "train_losses": train_losses,
                         "test_losses": test_losses, "time_list": time_list}, cp_file)

    @staticmethod
    def _load_checkpoint(filename: str):
        """
        Loads the state of the genetic algorithm from a checkpoint file.

        Args:
            filename (str): Path to the checkpoint file.

        Returns:
            dict: Loaded state from the checkpoint.
        """
        with open(filename, 'rb') as cp_file:
            state = pickle.load(cp_file)
        return state
