# import required libraries
import numpy as np
import time
from numba import njit, prange

'''
🔬 Performance Notes:
    + The do_sum_parallel function should outperform the others on multi-core CPUs, especially for large arrays.
    + Numerical differences can slightly increase due to parallel reduction (non-deterministic summation order).
    + Numba performs loop fusion and SIMD under fastmath + parallel, making this the most optimized variant (for speed, not accuracy).
'''

@njit(fastmath=False)
def do_sum(A):
    """
    Sequential sum of square roots with strict IEEE behavior.
    """
    acc = 0.
    for x in A:
        acc += np.sqrt(x)
    return acc

@njit(fastmath=True)
def do_sum_fast(A):
    """
    Sequential sum with relaxed math rules (fastmath).
    """
    acc = 0.
    for x in A:
        acc += np.sqrt(x)
    return acc

@njit(parallel=True, fastmath=True)
def do_sum_parallel(A):
    """
    Parallel sum using fastmath + parallel=True with prange.
    Accumulates in parallel with implicit reduction.
    """
    acc = 0.
    for i in prange(A.shape[0]):
        acc += np.sqrt(A[i])
    return acc

def benchmark(func, A, label):
    t0 = time.perf_counter()
    result = func(A)
    t1 = time.perf_counter()
    print(f"{label:<25} result = {result:.6f}, time = {(t1 - t0)*1000:.3f} ms")

if __name__ == "__main__":
    np.random.seed(0)
    A = np.random.rand(10_000_000).astype(np.float64)

    # Warm-up
    do_sum(A)
    do_sum_fast(A)
    do_sum_parallel(A)

    # Benchmark
    benchmark(do_sum, A, "Strict (fastmath=False)")
    benchmark(do_sum_fast, A, "Fast (fastmath=True)")
    benchmark(do_sum_parallel, A, "Fast + Parallel")

    # Compare numerical differences
    res_strict = do_sum(A)
    res_fast = do_sum_fast(A)
    res_parallel = do_sum_parallel(A)

    print(f"Δ(strict-fast):     {abs(res_strict - res_fast):.12e}")
    print(f"Δ(strict-parallel): {abs(res_strict - res_parallel):.12e}")
