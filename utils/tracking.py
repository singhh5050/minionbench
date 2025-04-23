import time
import functools

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        if isinstance(result, tuple):
            return result[0], {"latency": elapsed}
        return result, {"latency": elapsed}
    return wrapper
