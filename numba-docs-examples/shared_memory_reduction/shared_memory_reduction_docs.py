# import required libraries
import numpy as np
from numba import cuda
from numba.types import int32

# generate data
a = cuda.to_device(np.arange(1024))
nelem = len(a)

@cuda.jit

def array_sum(data):
    tid = cuda.threadIdx.x
    size = len(data)
    if tid < size:
        i = cuda.grid(1)

        # Declare an array in shared memory
        shr = cuda.shared.array(nelem, int32)
        shr[tid] = data[i]

        # Ensure writes to shared memory are visible
        # to all threads before reducing
        cuda.syncthreads()

        s = 1
        while s < cuda.blockDim.x:
            if tid % (2 * s) == 0:
                # Stride by `s` and add
                shr[tid] += shr[tid + s]
            s *= 2
            cuda.syncthreads()

        # After the loop, the zeroth  element contains the sum
        if tid == 0:
            data[tid] = shr[tid]

array_sum[1, nelem](a)
print(a[0])                  # 523776
print(sum(np.arange(1024)))  # 523776
