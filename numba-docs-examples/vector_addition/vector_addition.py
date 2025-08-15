# import required libraries
import numpy as np
import time
from numba import cuda

@cuda.jit
def vector_add_gpu(a, b, c):
    """
    CUDA kernel for vector addition: c = a + b
    Each thread computes one element of the result vector.
    """
    # Calculate global thread ID (like threadIdx.x + (blockIdx.x * blockDim.x))
    tid = cuda.grid(1)
    size = len(c)

    # Bounds check to ensure we don't access out-of-bounds memory
    if tid < size:
        c[tid] = a[tid] + b[tid]


def vector_add_cpu(a, b):
    """CPU version of vector addition for performance comparison."""
    return a + b


def main():
    # Vector size
    N = 100000
    print(f"Vector size: {N:,}")

    # Create random input vectors on CPU
    a_cpu = np.random.random(N).astype(np.float32)
    b_cpu = np.random.random(N).astype(np.float32)

    # CPU computation for comparison
    print("\n--- CPU Computation ---")
    start_time = time.time()
    c_cpu = vector_add_cpu(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.6f} seconds")

    # Transfer data to GPU
    print("\n--- GPU Computation ---")
    try:
        # Copy data to GPU device
        a_gpu = cuda.to_device(a_cpu)
        b_gpu = cuda.to_device(b_cpu)
        c_gpu = cuda.device_array_like(a_gpu)

        # Method 1: Using forall (automatic thread configuration)
        print("Method 1: Using forall (automatic configuration)")
        start_time = time.time()
        vector_add_gpu.forall(len(a_gpu))(a_gpu, b_gpu, c_gpu)
        cuda.synchronize()  # Wait for GPU to complete
        gpu_time_forall = time.time() - start_time
        c_result_forall = c_gpu.copy_to_host()
        print(f"GPU time (forall): {gpu_time_forall:.6f} seconds")

        # Reset result array for second method
        c_gpu = cuda.device_array_like(a_gpu)

        # Method 2: Manual block/thread configuration
        print("\nMethod 2: Manual block/thread configuration")
        nthreads = 256  # Threads per block (good for several warps per block)
        # Calculate blocks needed, ensuring we cover all elements
        nblocks = (N + nthreads - 1) // nthreads  # Ceiling division
        print(f"Blocks: {nblocks}, Threads per block: {nthreads}")
        print(f"Total threads: {nblocks * nthreads}")

        start_time = time.time()
        vector_add_gpu[nblocks, nthreads](a_gpu, b_gpu, c_gpu)
        cuda.synchronize()  # Wait for GPU to complete
        gpu_time_manual = time.time() - start_time
        c_result_manual = c_gpu.copy_to_host()
        print(f"GPU time (manual): {gpu_time_manual:.6f} seconds")

        # Verify results are correct
        print("\n--- Verification ---")
        forall_correct = np.allclose(c_cpu, c_result_forall, rtol=1e-5)
        manual_correct = np.allclose(c_cpu, c_result_manual, rtol=1e-5)

        print(f"Forall method correct: {forall_correct}")
        print(f"Manual method correct: {manual_correct}")

        # Performance comparison
        print("\n--- Performance Summary ---")
        print(f"CPU time:           {cpu_time:.6f} seconds")
        print(f"GPU time (forall):  {gpu_time_forall:.6f} seconds")
        print(f"GPU time (manual):  {gpu_time_manual:.6f} seconds")

        if gpu_time_forall > 0:
            speedup_forall = cpu_time / gpu_time_forall
            print(f"Speedup (forall):   {speedup_forall:.2f}x")

        if gpu_time_manual > 0:
            speedup_manual = cpu_time / gpu_time_manual
            print(f"Speedup (manual):   {speedup_manual:.2f}x")

    except Exception as e:
        print(f"CUDA Error: {e}")
        print("Make sure you have a CUDA-capable GPU and proper CUDA installation.")


if __name__ == "__main__":
    main()
