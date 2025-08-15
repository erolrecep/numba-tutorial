#!/usr/bin/env python3
"""
Simple NumPy Ufunc Usage in CUDA Kernels

This is a simplified version of the comprehensive ufunc example,
focusing on the basic concept of calling NumPy ufuncs within CUDA kernels.
"""

import numpy as np
import warnings
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

# Suppress performance warnings for educational examples
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


@cuda.jit
def ufunc_kernel(result, x):
    """
    Simple kernel calling a NumPy ufunc (sin) on each element.
    Demonstrates the basic concept of ufunc usage in CUDA.
    """
    idx = cuda.grid(1)
    if idx < len(x):
        # Call NumPy ufunc directly in CUDA kernel
        result[idx] = np.sin(x[idx])


def main():
    """
    Simple demonstration of NumPy ufunc usage in CUDA kernels.
    """
    print("🧮 Simple NumPy Ufunc in CUDA")
    print("=" * 40)

    try:
        # Create test data (same as original example)
        x = np.arange(10, dtype=np.float32) - 5
        print(f"Input array: {x}")

        # Allocate GPU memory
        d_x = cuda.to_device(x)
        d_result = cuda.device_array_like(x)

        # Configure grid (improved from original [1,1])
        block_size = 256
        num_blocks = (len(x) + block_size - 1) // block_size

        # Launch kernel
        ufunc_kernel[num_blocks, block_size](d_result, d_x)

        # Copy result back
        gpu_result = d_result.copy_to_host()

        # Compare with CPU
        cpu_result = np.sin(x)

        print(f"CPU sin(x):  {cpu_result}")
        print(f"GPU sin(x):  {gpu_result}")

        # Validate results
        try:
            np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-6)
            print("✅ Results match!")
            max_diff = np.max(np.abs(cpu_result - gpu_result))
            print(f"Maximum difference: {max_diff:.2e}")
        except AssertionError:
            print("❌ Results don't match!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have a CUDA-capable GPU and proper CUDA installation.")


if __name__ == "__main__":
    main()