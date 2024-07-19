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
import struct
from mycode.config import tracker

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)


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
        self.fitness_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

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

    def hash_individual(self, individual):
        """Create a hash for a GP individual (tree)."""
        m = hashlib.md5()
        for node in individual:
            if isinstance(node, gp.Primitive):
                m.update(node.name.encode())
            elif isinstance(node, gp.Terminal):
                if isinstance(node.value, float):
                    # Ensure consistent floating-point representation
                    m.update(bytearray(struct.pack("d", node.value)))
                else:
                    m.update(str(node.value).encode())
        return m.hexdigest()

    def hash_tree(self, trees):
        """Create a combined hash for multiple GP trees, ensuring all trees contribute to the final hash."""
        m = hashlib.md5()
        for tree in trees:
            tree_hash = self.hash_individual(tree)
            m.update(tree_hash.encode())
        return m.hexdigest()

    # def hash_tree(self, tree):
    #     """Create a hash for a GP tree."""
    #     m = hashlib.md5()
    #     for node in tree:
    #         if isinstance(node, gp.Primitive):
    #             m.update(node.name.encode())
    #         elif isinstance(node, gp.Terminal):
    #             m.update(str(node.value).encode())
    #     return m.hexdigest()

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
        # self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=self.tree_depth, max_=self.tree_depth)
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=2, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("select", tools.selTournament, k=15, tournsize=45)
        self.toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)
        self.toolbox.register("mate", self._crossover)

    @staticmethod
    @tracker.track_runtime
    def custom_ufunc(func, x):
        return np.array([func(*record) for record in x])

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
    def _evaluate_individual(self, individual, x: np.ndarray, y: np.ndarray, batch_size: int) -> tuple:
        # Check if we're dealing with a single tree or multiple trees for classification
        if self._num_classes == 1:
            individual_hash = self.hash_individual(individual)
        else:
            # Generate a hash for the list of trees in a multi-class scenario
            individual_hash = self.hash_tree(individual["ind"])

        # Use the hash to check the cache
        if individual_hash in self.fitness_cache:
            self.cache_hits += 1
            return self.fitness_cache[individual_hash]

        self.cache_misses += 1

        # Randomly select a batch of data
        # batch_indices = np.random.choice(len(x), size=batch_size, replace=False)
        x_batch = x
        y_batch = y

        # Evaluate the individual on the selected batch
        if self._num_classes == 1:
            func = self.toolbox.compile(expr=individual)
            predictions = GeneticAlgorithmModel._sigmoid(np.array([func(*record) for record in x_batch]))
        else:
            # For multi-class, evaluate each tree and collect predictions
            funcs = [self.toolbox.compile(expr=tree) for tree in individual["ind"]]
            predictions = np.array([[func(*record) for func in funcs] for record in x_batch])
            predictions = softmax(predictions, axis=1)
        fitness = log_loss(y_batch, predictions)

        # Update the cache
        self.fitness_cache[individual_hash] = (fitness,)
        return (fitness,)

    def run(self, lambd: int, max_generations: int, save_checkpoint_path: str, start_checkpoint: str = "",
            save_checkpoint: bool = False, batch_size: int = 1024) -> tuple:
        """
        Runs the genetic algorithm.

        Args:
            lambd (int): Number of offsprings in each generation.
            max_generations (int): Maximum number of generations.
            save_checkpoint_path (str): Path to save checkpoints.
            start_checkpoint (str, optional): Path to start checkpoint. Defaults to "".
            save_checkpoint (bool, optional): Whether to save checkpoints. Defaults to False.
            batch_size (int, optional): Size of the batch for evaluating individuals. Defaults to 32.

        Returns:
            tuple: Best individual, training losses, test losses, and time per generation.
        """

        stagnation_threshold = 100.0
        stagnation_count = 0
        full_eval_frequency = 15

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
                champion.fitness.values = self._evaluate_individual(champion, self.X_train, self.y_train, batch_size)
            else:
                champion = {"ind": [self.toolbox.individual() for _ in range(self._num_classes)],
                            "fitness": {"values": None}}
                champion_fitness = self._evaluate_individual(champion, self.X_train, self.y_train, batch_size)
                champion["fitness"]["values"] = champion_fitness[0]
            start_generation = 0
        test = []
        epsilon = 1e-8  # Small value for float comparisons
        old_champion = champion
        generations_without_improvement = 0
        for gen in tqdm(range(start_generation, max_generations), desc="Generations"):
            start_time = time.time()
            generations_without_improvement += 1

            # Generate a batch for this generation
            batch_indices = np.random.choice(len(self.X_train), size=batch_size, replace=False)
            x_batch = self.X_train[batch_indices]
            y_batch = self.y_train[batch_indices]

            # Re-evaluate champion on the new batch
            if self._num_classes == 1:
                champion.fitness.values = self._evaluate_individual(champion, x_batch, y_batch, batch_size)
            else:
                champion["fitness"]["values"] = self._evaluate_individual(champion, x_batch, y_batch, batch_size)[0]

            # Generate and evaluate candidates
            if self._num_classes == 1:
                candidates = [self.toolbox.clone(champion) for _ in range(lambd)]
                for i in range(0, len(candidates), 2):
                    if random.random() < 0.5:  # Crossover probability
                        candidates[i], candidates[i + 1] = self.toolbox.mate(candidates[i], candidates[i + 1])
                    self.toolbox.mutate(candidates[i])
                    self.toolbox.mutate(candidates[i + 1])
                    candidates[i].fitness.values = self._evaluate_individual(candidates[i], x_batch, y_batch, batch_size)
                    candidates[i + 1].fitness.values = self._evaluate_individual(candidates[i + 1], x_batch, y_batch, batch_size)
            else:
                candidates = [
                    {"ind": [self.toolbox.clone(tree) for tree in champion["ind"]], "fitness": {"values": None}}
                    for _ in range(lambd)
                ]
                for i in range(0, len(candidates), 2):
                    if random.random() < 0.5:  # Crossover probability
                        candidates[i], candidates[i + 1] = self.toolbox.mate(candidates[i], candidates[i + 1])
                    selected_tree = random.choice(candidates[i]["ind"])
                    self.toolbox.mutate(selected_tree)
                    selected_tree = random.choice(candidates[i + 1]["ind"])
                    self.toolbox.mutate(selected_tree)
                    candidates[i]["fitness"]["values"] = self._evaluate_individual(candidates[i], x_batch, y_batch, batch_size)[0]
                    candidates[i + 1]["fitness"]["values"] = self._evaluate_individual(candidates[i + 1], x_batch, y_batch, batch_size)[0]

            # Select the best candidate
            if self._num_classes == 1:
                best_candidate = tools.selBest(candidates, 1)[0]
                is_better = best_candidate.fitness.values[0] < champion.fitness.values[0] - epsilon
                if is_better:
                    champion = best_candidate
            else:
                sorted_list = sorted(candidates, key=lambda x: x["fitness"]["values"])
                best_candidate = sorted_list[0]
                is_better = best_candidate["fitness"]["values"] < champion["fitness"]["values"] - epsilon
                if is_better:
                    champion = best_candidate

            test.append(is_better)

            # Periodically evaluate on full dataset
            if gen % full_eval_frequency == 0:
                if self._num_classes == 1:
                    full_fitness = self._evaluate_individual_cv(champion, self.X_train, self.y_train)
                    champion.fitness.values = full_fitness
                else:
                    full_fitness = self._evaluate_individual_cv(champion, self.X_train, self.y_train)
                    champion["fitness"]["values"] = full_fitness[0]


            time_list.append(time.time() - start_time)

            train_loss = self._evaluate_individual(champion, self.X_train, self.y_train, batch_size)[0]
            print(f"Train Loss: {train_loss}")
            train_losses.append(train_loss)
            test_loss = self._evaluate_individual(champion, self.X_test, self.y_test, batch_size)[0]
            print(f"Test Loss: {test_loss}")
            test_losses.append(test_loss)

            if generations_without_improvement <250:
                stagnation_frequency = 15
            else:
                stagnation_frequency = 10

            # Check for stagnation
            if len(train_losses) >= stagnation_frequency:
                loss_diff = abs(train_losses[-1] - train_losses[-stagnation_frequency])
                if loss_diff < stagnation_threshold:
                    stagnation_count += 1
                else:
                    stagnation_count = 0

            # If stagnation is detected, perform a restart
            if stagnation_count >= stagnation_frequency:
                print("Stagnation detected. Performing restart.")
                candidates = self._perform_restart(champion, lambd)
                stagnation_count = 0

            if save_checkpoint:
                GeneticAlgorithmModel._save_checkpoint(champion, gen, train_losses, test_losses, time_list,
                                                       save_checkpoint_path)

        # Print the average execution time for each tracked function
        print(f"Average execution time for custom_ufunc: {tracker.get_average_time('custom_ufunc'):.4f} seconds")
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
        print(test)
        return champion, train_losses, test_losses, time_list, old_champion

    def make_predictions_with_threshold(self, individual, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if self._num_classes == 1:
            func = self.toolbox.compile(expr=individual)
            predictions_raw = GeneticAlgorithmModel.custom_ufunc(func, x)
            predictions = GeneticAlgorithmModel._sigmoid(predictions_raw)
            return (predictions > threshold).astype(int)
        else:
            funcs = [self.toolbox.compile(expr=tree) for tree in individual["ind"]]
            predictions_raw = np.array([GeneticAlgorithmModel.custom_ufunc(f, x) for f in funcs]).T
            predictions = softmax(predictions_raw, axis=1)
            return np.argmax(predictions, axis=1)

    def _evaluate_individual_cv(self, individual, x: np.ndarray, y: np.ndarray, n_splits=5):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fitness_scores = []

        for train_index, val_index in kf.split(x):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]

            if self._num_classes == 1:
                func = self.toolbox.compile(expr=individual)
                predictions = np.array([func(*record) for record in x_val])
            else:
                funcs = [self.toolbox.compile(expr=tree) for tree in individual["ind"]]
                predictions = np.array([[func(*record) for func in funcs] for record in x_val])
                predictions = softmax(predictions, axis=1)

            fitness = log_loss(y_val, predictions)
            fitness_scores.append(fitness)

        return (np.mean(fitness_scores),)

    def _perform_restart(self, champion, lambd):
        """
        Performs a restart by creating a new population with some of the best individuals
        and some newly generated individuals.
        """
        new_candidates = []

        # Keep the champion
        new_candidates.append(champion)

        # Add some mutated versions of the champion
        for _ in range(lambd // 3):
            if self._num_classes == 1:
                mutated = self.toolbox.clone(champion)
                self.toolbox.mutate(mutated)
                new_candidates.append(mutated)
            else:
                mutated = {"ind": [self.toolbox.clone(tree) for tree in champion["ind"]], "fitness": {"values": None}}
                selected_tree = random.choice(mutated["ind"])
                self.toolbox.mutate(selected_tree)
                new_candidates.append(mutated)

        # Add some completely new individuals
        for _ in range(lambd - len(new_candidates)):
            if self._num_classes == 1:
                new_ind = self.toolbox.individual()
                new_candidates.append(new_ind)
            else:
                new_ind = {"ind": [self.toolbox.individual() for _ in range(self._num_classes)], "fitness": {"values": None}}
                new_candidates.append(new_ind)

        return new_candidates

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

    def _crossover(self, parent1, parent2):
        """
        Performs crossover between two individuals.
        """
        if self._num_classes == 1:
            child1, child2 = gp.cxOnePoint(parent1, parent2)
            return child1, child2
        else:
            child1 = {"ind": [], "fitness": {"values": None}}
            child2 = {"ind": [], "fitness": {"values": None}}
            for tree1, tree2 in zip(parent1["ind"], parent2["ind"]):
                child1_tree, child2_tree = gp.cxOnePoint(tree1, tree2)
                child1["ind"].append(child1_tree)
                child2["ind"].append(child2_tree)
            return child1, child2
