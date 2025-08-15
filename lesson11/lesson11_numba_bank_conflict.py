import numpy as np
from numba import cuda, float32
import time

TPB = 32  # Threads per block
N = TPB   # One warp

@cuda.jit
def bank_conflict_kernel(output):
    shared = cuda.shared.array(shape=TPB, dtype=float32)
    tid = cuda.threadIdx.x

    # Load values to shared memory, stride access causes conflict
    shared[tid] = tid
    cuda.syncthreads()

    # Conflicting access: all threads access shared[tid * STRIDE]
    STRIDE = 2  # will cause bank conflict if stride > 1 and not coalesced
    val = shared[(tid * STRIDE) % TPB]
    output[tid] = val

@cuda.jit
def bank_no_conflict_kernel(output):
    shared = cuda.shared.array(shape=TPB + 1, dtype=float32)  # +1 padding
    tid = cuda.threadIdx.x

    shared[tid] = tid
    cuda.syncthreads()

    STRIDE = 2
    val = shared[(tid * STRIDE) % TPB]  # Access same pattern but no conflict due to padding
    output[tid] = val

# Run kernels
def run_and_time(kernel):
    out = np.zeros(N, dtype=np.float32)
    d_out = cuda.to_device(out)

    start = time.time()
    kernel[1, TPB](d_out)
    cuda.synchronize()
    end = time.time()

    result = d_out.copy_to_host()
    return result, end - start

# Run both
res_conflict, time_conflict = run_and_time(bank_conflict_kernel)
res_no_conflict, time_no_conflict = run_and_time(bank_no_conflict_kernel)

# Print results
print("Conflict Result:     ", res_conflict)
print("No Conflict Result:  ", res_no_conflict)
print(f"With Bank Conflict:  {time_conflict:.6f} s")
print(f"Without Conflict:    {time_no_conflict:.6f} s")
