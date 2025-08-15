# import required libraries
import numpy as np
import time
import warnings
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

# Suppress performance warnings for educational examples
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


@cuda.jit
def simple_ufunc_kernel(result, x):
    """
    Simple kernel calling a NumPy ufunc (sin) on each element.
    Each thread processes one element.
    """
    idx = cuda.grid(1)
    if idx < len(x):
        # Call NumPy ufunc directly in CUDA kernel
        result[idx] = np.sin(x[idx])


@cuda.jit
def multiple_ufuncs_kernel(result, x, y):
    """
    Kernel demonstrating multiple ufunc operations.
    Combines trigonometric, exponential, and arithmetic operations.
    """
    idx = cuda.grid(1)
    if idx < len(x):
        # Multiple ufunc operations in sequence
        sin_x = np.sin(x[idx])
        cos_y = np.cos(y[idx])
        exp_val = np.exp(x[idx] * 0.1)  # Scale to prevent overflow

        # Combine results using arithmetic ufuncs
        result[idx] = sin_x + cos_y * exp_val


@cuda.jit
def vectorized_ufunc_kernel(result, x):
    """
    Kernel using vectorized ufunc operations.
    Demonstrates calling ufuncs on array slices.
    """
    idx = cuda.grid(1)
    if idx < len(x):
        # Process multiple elements per thread (if array is large enough)
        start_idx = idx * 4
        end_idx = min(start_idx + 4, len(x))

        if start_idx < len(x):
            # Apply ufunc to a slice (vectorized operation)
            for i in range(start_idx, end_idx):
                result[i] = np.sqrt(np.abs(x[i]))


@cuda.jit
def conditional_ufunc_kernel(result, x):
    """
    Kernel with conditional ufunc usage.
    Applies different ufuncs based on input values.
    """
    idx = cuda.grid(1)
    if idx < len(x):
        val = x[idx]

        if val > 0:
            result[idx] = np.log(val + 1)  # log(x+1) for positive values
        elif val < 0:
            result[idx] = np.exp(val)      # exp(x) for negative values
        else:
            result[idx] = 1.0              # 1 for zero


def gpu_ufunc_computation(x, y=None, kernel_type='simple', block_size=256):
    """
    GPU computation using NumPy ufuncs in CUDA kernels.

    Args:
        x: Input array
        y: Second input array (for multi-input operations)
        kernel_type: Type of kernel to use
        block_size: CUDA block size

    Returns:
        Result array computed on GPU
    """
    n = len(x)

    # Ensure good GPU occupancy
    min_elements = 32 * block_size
    if n < min_elements:
        # Pad array for better GPU utilization
        x_padded = np.pad(x, (0, min_elements - n), mode='constant')
        if y is not None:
            y_padded = np.pad(y, (0, min_elements - n), mode='constant')
        else:
            y_padded = None
        n_padded = len(x_padded)
    else:
        x_padded = x
        y_padded = y
        n_padded = n

    # Transfer data to GPU
    d_x = cuda.to_device(x_padded.astype(np.float32))
    if y_padded is not None:
        d_y = cuda.to_device(y_padded.astype(np.float32))
    d_result = cuda.device_array(n_padded, dtype=np.float32)

    # Calculate grid configuration
    num_blocks = (n_padded + block_size - 1) // block_size

    # Launch appropriate kernel
    if kernel_type == 'simple':
        simple_ufunc_kernel[num_blocks, block_size](d_result, d_x)
    elif kernel_type == 'multiple':
        if y_padded is None:
            d_y = cuda.to_device(np.ones_like(x_padded, dtype=np.float32))
        multiple_ufuncs_kernel[num_blocks, block_size](d_result, d_x, d_y)
    elif kernel_type == 'vectorized':
        # Use fewer blocks for vectorized processing
        vec_blocks = (n_padded + block_size * 4 - 1) // (block_size * 4)
        vectorized_ufunc_kernel[vec_blocks, block_size](d_result, d_x)
    elif kernel_type == 'conditional':
        conditional_ufunc_kernel[num_blocks, block_size](d_result, d_x)

    # Copy result back and trim to original size
    result = d_result.copy_to_host()[:n]

    return result


def cpu_ufunc_computation(x, y=None, operation='simple'):
    """
    CPU reference implementation using NumPy ufuncs.

    Args:
        x: Input array
        y: Second input array (for multi-input operations)
        operation: Type of operation to perform

    Returns:
        Result array computed on CPU
    """
    if operation == 'simple':
        return np.sin(x)
    elif operation == 'multiple':
        if y is None:
            y = np.ones_like(x)
        return np.sin(x) + np.cos(y) * np.exp(x * 0.1)
    elif operation == 'vectorized':
        return np.sqrt(np.abs(x))
    elif operation == 'conditional':
        result = np.ones_like(x, dtype=np.float32)
        pos_mask = x > 0
        neg_mask = x < 0
        result[pos_mask] = np.log(x[pos_mask] + 1)
        result[neg_mask] = np.exp(x[neg_mask])
        return result

    return np.sin(x)  # Default


def validate_results(cpu_result, gpu_result, tolerance=1e-5):
    """
    Validate that CPU and GPU results match within tolerance.

    Args:
        cpu_result: CPU computed result
        gpu_result: GPU computed result
        tolerance: Numerical tolerance for comparison

    Returns:
        Boolean indicating if results match
    """
    try:
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=tolerance, atol=tolerance)
        return True
    except AssertionError:
        return False


def benchmark_ufunc_operations(x, y=None, num_trials=3):
    """
    Benchmark different ufunc operations on CPU vs GPU.

    Args:
        x: Input array
        y: Second input array (optional)
        num_trials: Number of timing trials

    Returns:
        Dictionary with timing results
    """
    operations = [
        ('simple', 'Simple sin(x)'),
        ('multiple', 'sin(x) + cos(y) * exp(0.1*x)'),
        ('vectorized', 'sqrt(abs(x))'),
        ('conditional', 'Conditional log/exp operations')
    ]

    print(f"\n⚡ NumPy Ufunc Benchmark (Array size: {len(x):,})")
    print("=" * 80)
    print(f"{'Operation':<25} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10} {'Valid':<8}")
    print("-" * 80)

    results = {}

    for op_type, description in operations:
        # CPU timing
        cpu_times = []
        for _ in range(num_trials):
            start_time = time.time()
            cpu_result = cpu_ufunc_computation(x, y, op_type)
            cpu_times.append((time.time() - start_time) * 1000)

        cpu_time_avg = np.mean(cpu_times)

        # GPU timing
        gpu_times = []
        for _ in range(num_trials):
            start_time = time.time()
            gpu_result = gpu_ufunc_computation(x, y, op_type)
            gpu_times.append((time.time() - start_time) * 1000)

        gpu_time_avg = np.mean(gpu_times)

        # Validation
        is_valid = validate_results(cpu_result, gpu_result)

        # Calculate speedup
        speedup = cpu_time_avg / gpu_time_avg if gpu_time_avg > 0 else 0

        # Store results
        results[op_type] = {
            'cpu_time': cpu_time_avg,
            'gpu_time': gpu_time_avg,
            'speedup': speedup,
            'valid': is_valid,
            'cpu_result': cpu_result,
            'gpu_result': gpu_result
        }

        # Display results
        valid_str = "✅ Yes" if is_valid else "❌ No"
        print(f"{description:<25} {cpu_time_avg:<12.3f} {gpu_time_avg:<12.3f} {speedup:<10.2f}x {valid_str:<8}")

    return results


def demonstrate_original_example():
    """
    Demonstrate the original example with improvements.
    """
    print("📋 Original Example (Enhanced):")
    print("-" * 40)

    # Original data
    x = np.arange(10, dtype=np.float32) - 5
    print(f"Input array: {x}")

    # CPU computation
    cpu_result = np.sin(x)
    print(f"CPU sin(x):  {cpu_result}")

    # GPU computation (improved)
    gpu_result = gpu_ufunc_computation(x, kernel_type='simple')
    print(f"GPU sin(x):  {gpu_result}")

    # Validation
    is_valid = validate_results(cpu_result, gpu_result)
    print(f"Results match: {'✅ Yes' if is_valid else '❌ No'}")

    if is_valid:
        max_diff = np.max(np.abs(cpu_result - gpu_result))
        print(f"Maximum difference: {max_diff:.2e}")


def demonstrate_ufunc_types():
    """
    Demonstrate different types of ufunc operations.
    """
    print("\n🔢 Different Ufunc Operations:")
    print("-" * 40)

    # Test data
    x = np.linspace(-2, 2, 16, dtype=np.float32)
    y = np.linspace(0, np.pi, 16, dtype=np.float32)

    operations = [
        ('simple', 'Trigonometric: sin(x)'),
        ('multiple', 'Combined: sin(x) + cos(y) * exp(0.1*x)'),
        ('vectorized', 'Mathematical: sqrt(abs(x))'),
        ('conditional', 'Conditional: log(x+1) or exp(x)')
    ]

    for op_type, description in operations:
        print(f"\n{description}:")

        cpu_result = cpu_ufunc_computation(x, y, op_type)
        gpu_result = gpu_ufunc_computation(x, y, op_type)

        is_valid = validate_results(cpu_result, gpu_result)

        print(f"  Input sample:  {x[:4]} ...")
        print(f"  CPU result:    {cpu_result[:4]} ...")
        print(f"  GPU result:    {gpu_result[:4]} ...")
        print(f"  Match: {'✅' if is_valid else '❌'}")


def analyze_ufunc_performance():
    """
    Analyze performance characteristics of ufunc operations.
    """
    print("\n📈 Performance Analysis:")
    print("-" * 40)

    # Test different array sizes
    sizes = [1000, 10000, 100000]

    for size in sizes:
        print(f"\nArray size: {size:,} elements")

        # Generate test data
        x = np.random.randn(size).astype(np.float32)
        y = np.random.randn(size).astype(np.float32)

        # Run benchmark
        results = benchmark_ufunc_operations(x, y, num_trials=1)

        # Find best performing operation
        best_op = max(results.keys(), key=lambda k: results[k]['speedup'])
        best_speedup = results[best_op]['speedup']

        print(f"Best GPU speedup: {best_speedup:.2f}x ({best_op})")


def main():
    """
    Main function demonstrating NumPy ufunc usage in CUDA kernels.
    """
    print("🧮 CUDA NumPy Ufunc Integration")
    print("=" * 60)

    try:
        # Demonstrate original example
        demonstrate_original_example()

        # Show different ufunc types
        demonstrate_ufunc_types()

        # Performance analysis
        analyze_ufunc_performance()

        print("\n🎓 Key Learning Points:")
        print("-" * 30)
        print("1. NumPy ufuncs can be called directly in CUDA kernels")
        print("2. GPU memory management is crucial for performance")
        print("3. Proper grid configuration improves GPU utilization")
        print("4. Different ufuncs have varying performance characteristics")
        print("5. Validation ensures correctness across platforms")

        print("\n🔧 Technical Details:")
        print("-" * 20)
        print("• Ufuncs are automatically vectorized in CUDA kernels")
        print("• Memory coalescing improves bandwidth utilization")
        print("• Thread divergence can affect conditional operations")
        print("• Proper data types (float32) optimize GPU performance")
        print("• Error handling ensures robust execution")

        print("\n✨ Supported Ufuncs:")
        print("-" * 18)
        print("• Trigonometric: sin, cos, tan, arcsin, arccos, arctan")
        print("• Exponential: exp, log, log10, sqrt, power")
        print("• Arithmetic: add, subtract, multiply, divide")
        print("• Comparison: greater, less, equal, not_equal")
        print("• Logical: logical_and, logical_or, logical_not")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("Make sure you have a CUDA-capable GPU and proper CUDA installation.")


if __name__ == "__main__":
    main()
