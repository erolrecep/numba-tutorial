import numpy as np
from numba import vectorize, float64
import time

# Vectorized function: squares each element
@vectorize([float64(float64)])
def square(x):
    return x * x

# Input data
arr = np.linspace(0, 10, 10_000_000, dtype=np.float64)

# Pure NumPy timing for comparison
start = time.perf_counter()
res_np = np.square(arr)
end = time.perf_counter()
print(f"NumPy square time: {end - start:.4f} s")

# Numba vectorize timing
start = time.perf_counter()
res_nb = square(arr)
end = time.perf_counter()
print(f"Numba vectorize time: {end - start:.4f} s")

# Verify correctness
assert np.allclose(res_np, res_nb)

