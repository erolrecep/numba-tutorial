import numpy as np
import time
from numba import jit, njit, prange

def stable_sigmoid(x):
    out = np.empty_like(x)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1 + exp_x)
    return out

def pure_numpy_logistic_regression(Y, X, w, iterations):
    for i in range(iterations):
        z = np.dot(X, w)
        pred = stable_sigmoid(Y * z)
        w -= np.dot((pred - 1.0) * Y, X)
    return w

@njit
def njit_logistic_regression(Y, X, w, iterations):
    for i in range(iterations):
        z = np.dot(X, w)
        pred = 1.0 / (1.0 + np.exp(-Y * z))
        w -= np.dot((pred - 1.0) * Y, X)
    return w

@njit(parallel=True)
def njit_parallel_logistic_regression(Y, X, w, iterations):
    for i in range(iterations):
        z = np.dot(X, w)
        pred = 1.0 / (1.0 + np.exp(-Y * z))
        w -= np.dot((pred - 1.0) * Y, X)
    return w

@jit
def jit_logistic_regression(Y, X, w, iterations):
    for i in range(iterations):
        z = np.dot(X, w)
        pred = 1.0 / (1.0 + np.exp(-Y * z))
        w -= np.dot((pred - 1.0) * Y, X)
    return w

@jit(nopython=True)
def jit_nopython_logistic_regression(Y, X, w, iterations):
    for i in range(iterations):
        z = np.dot(X, w)
        pred = 1.0 / (1.0 + np.exp(-Y * z))
        w -= np.dot((pred - 1.0) * Y, X)
    return w

# Setup
np.random.seed(0)
X = np.random.randn(1000, 100)
Y = np.random.choice([-1, 1], size=1000).astype(np.float64)
w = np.zeros(100)
iterations = 10

def benchmark(func, Y, X, w, iterations, label):
    w_copy = w.copy()
    start = time.perf_counter()
    func(Y, X, w_copy, iterations)
    elapsed = time.perf_counter() - start
    print(f"{label:<30}: {elapsed:.6f} seconds")

print("Benchmark results:")
benchmark(pure_numpy_logistic_regression, Y, X, w, iterations, "Pure NumPy")
benchmark(njit_logistic_regression, Y, X, w, iterations, "Numba @njit")
benchmark(njit_parallel_logistic_regression, Y, X, w, iterations, "Numba @njit(parallel=True)")
benchmark(jit_logistic_regression, Y, X, w, iterations, "Numba @jit")
benchmark(jit_nopython_logistic_regression, Y, X, w, iterations, "Numba @jit(nopython=True)")
