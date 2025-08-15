# import required libraries
import numpy as np
import time
from numba import njit

'''
🧠 Notes:
    fastmath=True can yield faster execution by allowing:
    + Floating-point reassociation
    + Vectorization (e.g., AVX)
    + Ignoring IEEE edge cases (NaNs, +/-0, infinities)
    + Expect minor differences (ULPs) in floating-point sum results.
    + Best for performance-critical workloads where bit-exact reproducibility is not required.
'''

@njit(fastmath=False)
def do_sum(A):
    """
    Sum of square roots using strict IEEE-754 floating-point behavior.
    Reassociation and unsafe math optimizations are disallowed.
    """
    acc = 0.
    for x in A:
        acc += np.sqrt(x)
    return acc

@njit(fastmath=True)
def do_sum_fast(A):
    """
    Sum of square roots using fastmath: allows reassociation, vectorization, and fused operations.
    May yield faster results but with minor numerical differences.
    """
    acc = 0.
    for x in A:
        acc += np.sqrt(x)
    return acc

def benchmark(func, A, label):
    t0 = time.perf_counter()
    result = func(A)
    t1 = time.perf_counter()
    print(f"{label:<20} result = {result:.6f}, time = {(t1 - t0)*1000:.3f} ms")

if __name__ == "__main__":
    np.random.seed(0)
    A = np.random.rand(10_000_000).astype(np.float64)

    # Warm-up JIT
    do_sum(A)
    do_sum_fast(A)

    # Benchmark
    benchmark(do_sum, A, "Strict (fastmath=False)")
    benchmark(do_sum_fast, A, "Fast (fastmath=True)")

    # Compare numerical difference
    res_strict = do_sum(A)
    res_fast = do_sum_fast(A)
    print(f"Difference: {abs(res_strict - res_fast):.12e}")
