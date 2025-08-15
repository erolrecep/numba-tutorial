import numpy as np
import time
from numba import njit, prange

# Pure Python
def sum_of_squares_py(arr):
    total = 0.0
    for i in range(len(arr)):
        total += arr[i] * arr[i]
    return total

# Numba JIT (single-threaded)
@njit
def sum_of_squares_numba(arr):
    total = 0.0
    for i in range(len(arr)):
        total += arr[i] * arr[i]
    return total

# Numba JIT + Parallel
@njit(parallel=True)
def sum_of_squares_parallel(arr):
    total = 0.0
    for i in prange(len(arr)):
        total += arr[i] * arr[i]
    return total

# Benchmark utility
def benchmark(fn, arr):
    fn(arr)  # warmup
    start = time.perf_counter()
    result = fn(arr)
    end = time.perf_counter()
    return result, end - start

# Input array
arr = np.random.rand(20_000_000).astype(np.float64)

# Run benchmarks
res_py, t_py = benchmark(sum_of_squares_py, arr)
res_njit, t_njit = benchmark(sum_of_squares_numba, arr)
res_par, t_par = benchmark(sum_of_squares_parallel, arr)

# Check correctness
assert np.allclose(res_py, res_njit)
assert np.allclose(res_py, res_par)

# Report
print(f"Pure Python       : {t_py:.4f} s")
print(f"Numba (njit)      : {t_njit:.4f} s")
print(f"Numba (parallel)  : {t_par:.4f} s")

