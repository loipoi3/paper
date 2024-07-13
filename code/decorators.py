import time
from functools import wraps


class ExecutionTimeTracker:
    def __init__(self):
        self.execution_times = {}

    def track_runtime(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            runtime = end_time - start_time
            if func.__name__ not in self.execution_times:
                self.execution_times[func.__name__] = []
            self.execution_times[func.__name__].append(runtime)
            return result
        return wrapper

    def get_average_time(self, func_name):
        if func_name not in self.execution_times or not self.execution_times[func_name]:
            return 0
        return sum(self.execution_times[func_name])
