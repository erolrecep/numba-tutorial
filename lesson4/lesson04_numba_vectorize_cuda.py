import numpy as np
from numba import vectorize, float64
import time

@vectorize([float64(float64)], target='cuda')
def square_cuda(x):
    return x * x

arr = np.linspace(0, 10, 10_000_000, dtype=np.float64)

# Warmup to compile and transfer to GPU
square_cuda(arr[:10])

start = time.perf_counter()
res_gpu = square_cuda(arr)
end = time.perf_counter()

print(f"Numba CUDA vectorize time: {end - start:.4f} s")

# Verify correctness against CPU numpy
res_cpu = np.square(arr)
assert np.allclose(res_cpu, res_gpu)

