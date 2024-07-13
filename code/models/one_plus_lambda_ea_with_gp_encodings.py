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
import hashlib


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
        self.fitness_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def hash_individual(self, individual):
        if isinstance(individual, gp.PrimitiveTree):
            return self.hash_tree(individual)
        elif isinstance(individual, dict):  # For multi-tree individuals
            return tuple(self.hash_tree(tree) for tree in individual)
        else:
            raise TypeError(f"Unsupported individual type: {type(individual)}")


    def hash_tree(self, tree):
        """Create a hash for a GP tree."""
        m = hashlib.md5()
        for node in tree:
            if isinstance(node, gp.Primitive):
                m.update(node.name.encode())
            elif isinstance(node, gp.Terminal):
                m.update(str(node.value).encode())
        return m.hexdigest()

    def _setup_primitives(self):
        """
        Sets up the primitive set for the genetic programming algorithm using a mapping approach to improve readability and efficiency.
        """
        primitive_operations = {
            "add": (operator.add, 2),
            "sub": (operator.sub, 2),
            "mul": (operator.mul, 2),
            "_safe_div": (self._safe_div, 2),
            "min": (min, 2),
            "max": (max, 2),
            "hypot": (math.hypot, 2),
            "logaddexp": (np.logaddexp, 2),
            "_safe_atan2": (self._safe_atan2, 2),
            "_float_lt": (self._float_lt, 2),
            "_float_gt": (self._float_gt, 2),
            "_float_ge": (self._float_ge, 2),
            "_float_le": (self._float_le, 2),
            "_safe_fmod": (self._safe_fmod, 2)
        }

        # Add primitives based on the provided set
        for primitive_name, (func, arity) in primitive_operations.items():
            if primitive_name in self._primitive_set:
                self.pset.addPrimitive(func, arity)

        # Terminal values and their mappings
        terminal_values = {
            "Constant_0": 0,
            "Constant_1": 1,
            "Constant_minus_1": -1,
            "Pi": math.pi,
            "E": math.e
        }

        # Add terminals based on the provided set
        for terminal_name, value in terminal_values.items():
            if terminal_name in self._terminal_set:
                self.pset.addTerminal(value, terminal_name)

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
    def custom_ufunc(func, *args):
        n_rows = len(args[0])
        return np.array([func(*[arg[i] for arg in args]) for i in range(n_rows)])

    @staticmethod
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

    def _evaluate_individual(self, individual, x: np.ndarray, y: np.ndarray) -> tuple:
        # Create a hash for the individual
        individual_hash = self.hash_individual(individual)

        # Create hash of x and y
        x_hash = hash(x.tobytes())
        y_hash = hash(y.tobytes())

        # Check if this individual has been evaluated before
        cache_key = (individual_hash, x_hash, y_hash)
        if cache_key in self.fitness_cache:
            self.cache_hits += 1
            return self.fitness_cache[cache_key]

        self.cache_misses += 1

        # If not in cache, evaluate
        if self._num_classes == 1:
            if isinstance(individual, dict):
                expr = individual['ind']
            else:
                expr = individual
            func = self.toolbox.compile(expr=expr)
            predictions = np.array([GeneticAlgorithmModel._sigmoid(func(*record)) for record in x])
        else:
            if isinstance(individual, dict):
                trees = individual['ind']
            else:
                trees = individual
            funcs = [self.toolbox.compile(expr=tree) for tree in trees]
            predictions = np.array([[func(*record) for func in funcs] for record in x])
            predictions = softmax(predictions, axis=1)

        fitness = log_loss(y, predictions)

        # Store in cache
        self.fitness_cache[cache_key] = (fitness,)

        return (fitness,)

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

        return champion, train_losses, test_losses, time_list

    # @njit(object_mode=True)
    # @jit(nopython=True)
    def make_predictions_with_threshold(self, individual, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if self._num_classes == 1:
            func = self.toolbox.compile(expr=individual)
            predictions_raw = GeneticAlgorithmModel.custom_ufunc(func, *[x[:, i] for i in range(x.shape[1])])
            predictions = GeneticAlgorithmModel._sigmoid(predictions_raw)
            return (predictions > threshold).astype(int)
        else:
            funcs = [self.toolbox.compile(expr=tree) for tree in individual["ind"]]
            predictions_raw = np.array([GeneticAlgorithmModel.custom_ufunc(f, *[x[:, i] for i in range(x.shape[1])]) for f in funcs]).T
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
