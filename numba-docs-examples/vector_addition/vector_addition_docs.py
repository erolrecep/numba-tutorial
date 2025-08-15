# import required libraries
import numpy as np
from numba import cuda

@cuda.jit
def f(a, b, c):
    # like threadIdx.x + (blockIdx.x * blockDim.x)
    tid = cuda.grid(1)
    size = len(c)

    if tid < size:
        c[tid] = a[tid] + b[tid]


N = 100000
a = cuda.to_device(np.random.random(N))
b = cuda.to_device(np.random.random(N))
c = cuda.device_array_like(a)

# f.forall(len(a))(a, b, c)
# print(c.copy_to_host())

# Enough threads per block for several warps per block
nthreads = 256

# Enough blocks to cover the entire vector depending on its length
nblocks = (len(a) // nthreads) + 1
f[nblocks, nthreads](a, b, c)
print(c.copy_to_host())
