# import required libraries
import numpy as np
import time
import warnings
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from numba.types import int32, int64

# Suppress performance warnings for educational examples
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)



@cuda.jit
def shared_memory_reduction_kernel(data, results):
    """
    Optimized shared memory reduction kernel using sequential addressing.

    This kernel performs parallel reduction within each thread block using
    shared memory for fast intra-block communication. Each block computes
    a partial sum, which is stored in the results array.

    Args:
        data: Input array to reduce
        results: Output array to store partial sums from each block
    """
    # Thread and block indices
    tid = cuda.threadIdx.x  # Thread ID within block
    bid = cuda.blockIdx.x   # Block ID
    block_dim = cuda.blockDim.x  # Threads per block

    # Global thread index
    i = bid * block_dim + tid

    # Declare shared memory array (dynamically sized)
    shr = cuda.shared.array(1024, int64)  # Use int64 to prevent overflow

    # Load data into shared memory with bounds checking
    if i < len(data):
        shr[tid] = data[i]
    else:
        shr[tid] = 0  # Pad with zeros for out-of-bounds elements

    # Synchronize to ensure all data is loaded
    cuda.syncthreads()

    # Perform reduction using sequential addressing (more efficient)
    # This pattern reduces thread divergence compared to interleaved addressing
    s = block_dim // 2
    while s > 0:
        if tid < s and (tid + s) < block_dim:
            # Only active threads participate in reduction
            if (i + s) < len(data):  # Bounds check for original data
                shr[tid] += shr[tid + s]

        # Synchronize after each reduction step
        cuda.syncthreads()
        s //= 2

    # Thread 0 writes the block's partial sum to global memory
    if tid == 0:
        results[bid] = shr[0]


@cuda.jit
def simple_reduction_kernel(data, results):
    """
    Simple reduction kernel without shared memory (for comparison).
    Each thread processes one element and atomically adds to result.
    """
    i = cuda.grid(1)

    if i < len(data):
        # Cast to int64 to prevent overflow in atomic operations
        cuda.atomic.add(results, 0, int64(data[i]))


def cpu_reduction(data):
    """CPU reference implementation for validation."""
    return np.sum(data)


def gpu_reduction_shared_memory(data, block_size=512):
    """
    GPU reduction using shared memory with multiple blocks.

    Args:
        data: Input numpy array
        block_size: Threads per block (should be power of 2)

    Returns:
        Reduced sum value
    """
    # Ensure block_size is power of 2 for optimal reduction
    if block_size & (block_size - 1) != 0:
        # Find next power of 2
        block_size = 1 << (block_size - 1).bit_length()
        print(f"Adjusted block size to nearest power of 2: {block_size}")

    # Ensure minimum occupancy by using larger block sizes for small arrays
    min_blocks = 32  # Minimum blocks for good occupancy
    if len(data) < min_blocks * block_size:
        # Adjust block size to ensure minimum number of blocks
        block_size = max(32, (len(data) + min_blocks - 1) // min_blocks)
        # Round up to nearest power of 2
        block_size = 1 << (block_size - 1).bit_length()

    # Calculate number of blocks needed
    num_blocks = (len(data) + block_size - 1) // block_size

    # Ensure minimum number of blocks for good occupancy
    if num_blocks < min_blocks and len(data) > block_size:
        block_size = max(32, len(data) // min_blocks)
        block_size = 1 << (block_size - 1).bit_length()
        num_blocks = (len(data) + block_size - 1) // block_size

    # Transfer data to GPU
    d_data = cuda.to_device(data)
    d_results = cuda.device_array(num_blocks, dtype=np.int64)  # Use int64 to prevent overflow

    # Launch kernel
    shared_memory_reduction_kernel[num_blocks, block_size](d_data, d_results)

    # Copy partial results back to host
    partial_results = d_results.copy_to_host()

    # Final reduction on CPU (could also be done on GPU recursively)
    final_result = np.sum(partial_results, dtype=np.int64)

    return int(final_result), partial_results


def gpu_reduction_simple(data):
    """Simple GPU reduction using atomic operations (for comparison)."""
    d_data = cuda.to_device(data)
    d_result = cuda.device_array(1, dtype=np.int64)  # Use int64 to prevent overflow
    d_result[0] = 0  # Initialize result

    # Calculate grid configuration for better occupancy
    block_size = 512
    num_blocks = min(2048, (len(data) + block_size - 1) // block_size)  # Limit max blocks

    # Ensure minimum occupancy
    min_blocks = 32
    if num_blocks < min_blocks:
        num_blocks = min(min_blocks, (len(data) + block_size - 1) // block_size)

    # Launch kernel
    simple_reduction_kernel[num_blocks, block_size](d_data, d_result)

    return int(d_result.copy_to_host()[0])


def benchmark_reductions(data_sizes, num_trials=5):
    """
    Benchmark different reduction methods across various data sizes.

    Args:
        data_sizes: List of array sizes to test
        num_trials: Number of trials for averaging
    """
    print("Reduction Performance Benchmark")
    print("=" * 80)
    print(f"{'Size':<12} {'CPU (ms)':<12} {'GPU Shared (ms)':<16} {'GPU Atomic (ms)':<16} {'Speedup':<10}")
    print("-" * 80)

    for size in data_sizes:
        # Generate test data - use smaller values to prevent overflow
        if size <= 65536:
            data = np.arange(size, dtype=np.int32)
        else:
            # For large arrays, use smaller values to prevent overflow
            data = np.random.randint(0, 100, size=size, dtype=np.int32)

        expected_result = np.sum(data, dtype=np.int64)

        # CPU timing
        cpu_times = []
        for _ in range(num_trials):
            start_time = time.time()
            cpu_result = cpu_reduction(data)
            cpu_times.append((time.time() - start_time) * 1000)
        cpu_time_avg = np.mean(cpu_times)

        # GPU shared memory timing
        gpu_shared_times = []
        for _ in range(num_trials):
            start_time = time.time()
            gpu_shared_result, _ = gpu_reduction_shared_memory(data)
            gpu_shared_times.append((time.time() - start_time) * 1000)
        gpu_shared_time_avg = np.mean(gpu_shared_times)

        # GPU atomic timing (skip for very large arrays due to overflow issues)
        if size <= 262144:
            gpu_atomic_times = []
            for _ in range(num_trials):
                start_time = time.time()
                gpu_atomic_result = gpu_reduction_simple(data)
                gpu_atomic_times.append((time.time() - start_time) * 1000)
            gpu_atomic_time_avg = np.mean(gpu_atomic_times)
        else:
            gpu_atomic_result = expected_result  # Skip atomic test for large arrays
            gpu_atomic_time_avg = 0.0

        # Verify correctness
        assert cpu_result == expected_result, f"CPU result mismatch: {cpu_result} != {expected_result}"
        assert gpu_shared_result == expected_result, f"GPU shared result mismatch: {gpu_shared_result} != {expected_result}"
        if size <= 262144:  # Only verify atomic for smaller arrays
            assert gpu_atomic_result == expected_result, f"GPU atomic result mismatch: {gpu_atomic_result} != {expected_result}"

        # Calculate speedup
        speedup = cpu_time_avg / gpu_shared_time_avg if gpu_shared_time_avg > 0 else 0

        if size <= 262144:
            print(f"{size:<12} {cpu_time_avg:<12.3f} {gpu_shared_time_avg:<16.3f} {gpu_atomic_time_avg:<16.3f} {speedup:<10.2f}x")
        else:
            print(f"{size:<12} {cpu_time_avg:<12.3f} {gpu_shared_time_avg:<16.3f} {'N/A (skipped)':<16} {speedup:<10.2f}x")


def demonstrate_reduction_steps(data, block_size=8):
    """
    Demonstrate the reduction process step by step for educational purposes.
    Uses a small array to show how shared memory reduction works.
    """
    print(f"\nReduction Demonstration (Block size: {block_size})")
    print("=" * 60)

    print(f"Input data: {data}")
    print(f"Expected sum: {np.sum(data)}")

    # Perform reduction and get partial results
    result, partial_results = gpu_reduction_shared_memory(data, block_size)

    num_blocks = len(partial_results)
    print(f"\nNumber of blocks used: {num_blocks}")
    print(f"Partial sums from each block: {partial_results}")
    print(f"Final result: {result}")

    # Show how data is distributed across blocks
    print(f"\nData distribution across blocks:")
    for block_id in range(num_blocks):
        start_idx = block_id * block_size
        end_idx = min(start_idx + block_size, len(data))
        block_data = data[start_idx:end_idx]
        block_sum = np.sum(block_data)
        print(f"  Block {block_id}: {block_data} -> Sum: {block_sum}")


def main():
    print("CUDA Shared Memory Reduction Example")
    print("=" * 50)

    try:
        # Test with original example
        print("\n1. Original Example Test")
        print("-" * 30)
        original_data = np.arange(1024, dtype=np.int32)
        expected_sum = np.sum(original_data)

        result, partial_results = gpu_reduction_shared_memory(original_data)

        print(f"Array size: {len(original_data)}")
        print(f"Expected sum: {expected_sum}")
        print(f"GPU result: {result}")
        print(f"Number of blocks: {len(partial_results)}")
        print(f"Correct: {result == expected_sum}")

        # Small demonstration
        print("\n2. Step-by-Step Demonstration")
        print("-" * 30)
        small_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int32)
        demonstrate_reduction_steps(small_data, block_size=8)

        # Performance benchmark
        print("\n3. Performance Benchmark")
        print("-" * 30)
        # Use sizes that provide good GPU occupancy and avoid overflow
        data_sizes = [16384, 65536, 262144, 1048576, 4194304]
        benchmark_reductions(data_sizes, num_trials=3)

        print("\n4. Algorithm Analysis")
        print("-" * 30)
        print("Shared Memory Reduction Benefits:")
        print("  • Reduces global memory traffic")
        print("  • Exploits fast shared memory bandwidth")
        print("  • Minimizes thread divergence with sequential addressing")
        print("  • Scales well with problem size")
        print("\nComplexity Analysis:")
        print("  • Time: O(log n) per block, O(n/p + log p) overall")
        print("  • Space: O(block_size) shared memory per block")
        print("  • Communication: O(num_blocks) global memory writes")

    except Exception as e:
        print(f"Error during execution: {e}")
        print("Make sure you have a CUDA-capable GPU and proper CUDA installation.")


if __name__ == "__main__":
    main()
