import math
import threading
from timeit import repeat

import numpy as np
from numba import jit

# Configuration
nthreads = 4
size = 10**6


def func_np(a, b):
    """Control function using NumPy."""
    return np.exp(2.1 * a + 3.2 * b)


@jit('void(double[:], double[:], double[:])', nopython=True, nogil=True)
def inner_func_nb(result, a, b):
    """
    Numba-compiled function that releases the GIL.
    Computes: result[i] = exp(2.1 * a[i] + 3.2 * b[i])
    """
    for i in range(len(result)):
        result[i] = math.exp(2.1 * a[i] + 3.2 * b[i])


def timefunc(correct, s, func, *args, **kwargs):
    """
    Benchmark a function and print out its runtime.

    Parameters:
    correct: expected result for validation (None to skip validation)
    s: description string for the benchmark
    func: function to benchmark
    """
    print(s.ljust(20), end=" ")

    # Make sure the function is compiled before the benchmark starts
    res = func(*args, **kwargs)

    if correct is not None:
        assert np.allclose(res, correct), (res, correct)

    # Time the function execution
    runtime = min(repeat(lambda: func(*args, **kwargs), number=5, repeat=2)) * 1000
    print('{:>5.0f} ms'.format(runtime))

    return res


def make_singlethread(inner_func):
    """
    Create a single-threaded wrapper for the given function.
    """
    def func(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        inner_func(result, *args)
        return result
    return func


def make_multithread(inner_func, numthreads):
    """
    Create a multi-threaded wrapper for the given function.
    Splits input arrays into equal-sized chunks and processes them in parallel.

    Parameters:
    inner_func: the Numba function to parallelize (must have nogil=True)
    numthreads: number of threads to use
    """
    def func_mt(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        args = (result,) + args

        # Calculate chunk size for each thread
        chunklen = (length + numthreads - 1) // numthreads

        # Create argument tuples for each input chunk
        chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in args]
                  for i in range(numthreads)]

        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        return result
    return func_mt


# Create single-threaded and multi-threaded versions
func_nb = make_singlethread(inner_func_nb)
func_nb_mt = make_multithread(inner_func_nb, nthreads)

# Generate test data
a = np.random.rand(size)
b = np.random.rand(size)

# Run benchmarks
print("Benchmarking different implementations:")
correct = timefunc(None, "numpy (1 thread)", func_np, a, b)
timefunc(correct, "numba (1 thread)", func_nb, a, b)
timefunc(correct, "numba (%d threads)" % nthreads, func_nb_mt, a, b)

