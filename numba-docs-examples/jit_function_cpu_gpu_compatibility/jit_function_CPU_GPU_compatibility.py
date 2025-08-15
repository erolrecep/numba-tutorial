# import required libraries
import numpy as np
import time
import warnings
import math
from numba import jit, cuda
from numba.core.errors import NumbaPerformanceWarning

# Suppress performance warnings for educational examples
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


@jit
def business_logic(x, y, z):
    """
    A mathematical function that demonstrates CPU/GPU compatibility.

    This function can be called from both CPU (@jit) and GPU (@cuda.jit) contexts
    without modification, showcasing Numba's unified compilation model.

    Formula: 4 * z * (2 * x - (4 * y) / 2 * π)

    Args:
        x, y, z: Numerical inputs (float or int)

    Returns:
        Computed result as float
    """
    return 4 * z * (2 * x - (4 * y) / 2 * math.pi)


@jit
def polynomial_logic(x, y, z):
    """
    Simpler mathematical function that works on both CPU and GPU.

    Uses only basic arithmetic operations that are fully supported.
    Formula: x^2 + y^2 + z^2 + 2*x*y + 3*z
    """
    return x*x + y*y + z*z + 2*x*y + 3*z


@cuda.jit
def gpu_compute_kernel(results, x_arr, y_arr, z_arr, func_selector=0):
    """
    GPU kernel that can call different business logic functions.

    Args:
        results: Output array for computed results
        x_arr, y_arr, z_arr: Input arrays
        func_selector: 0 for business_logic, 1 for polynomial_logic
    """
    tid = cuda.grid(1)

    if tid < len(x_arr):
        if func_selector == 0:
            # Call the simple business logic function
            results[tid] = business_logic(x_arr[tid], y_arr[tid], z_arr[tid])
        else:
            # Call the polynomial logic function
            results[tid] = polynomial_logic(x_arr[tid], y_arr[tid], z_arr[tid])


def cpu_compute_vectorized(x_arr, y_arr, z_arr, func_type='simple'):
    """
    CPU vectorized computation using NumPy and Numba JIT.

    Args:
        x_arr, y_arr, z_arr: Input arrays
        func_type: 'simple' or 'polynomial'

    Returns:
        Computed results array
    """
    n = len(x_arr)
    results = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if func_type == 'simple':
            results[i] = business_logic(x_arr[i], y_arr[i], z_arr[i])
        else:
            results[i] = polynomial_logic(x_arr[i], y_arr[i], z_arr[i])

    return results


def gpu_compute_wrapper(x_arr, y_arr, z_arr, func_type='simple', block_size=256):
    """
    GPU computation wrapper with proper memory management.

    Args:
        x_arr, y_arr, z_arr: Input arrays (NumPy)
        func_type: 'simple' or 'polynomial'
        block_size: CUDA block size

    Returns:
        Computed results array
    """
    n = len(x_arr)

    # Transfer data to GPU
    d_x = cuda.to_device(x_arr.astype(np.float64))
    d_y = cuda.to_device(y_arr.astype(np.float64))
    d_z = cuda.to_device(z_arr.astype(np.float64))
    d_results = cuda.device_array(n, dtype=np.float64)

    # Calculate grid configuration
    num_blocks = (n + block_size - 1) // block_size

    # Launch kernel
    func_selector = 0 if func_type == 'simple' else 1
    gpu_compute_kernel[num_blocks, block_size](d_results, d_x, d_y, d_z, func_selector)

    # Copy results back to host
    results = d_results.copy_to_host()

    return results


def generate_test_data(size=1000):
    """
    Generate test data with various numerical ranges and edge cases.

    Args:
        size: Number of data points to generate

    Returns:
        Tuple of (x_arr, y_arr, z_arr) NumPy arrays
    """
    np.random.seed(42)  # For reproducible results

    # Generate diverse test data
    x_arr = np.random.uniform(-10, 10, size)
    y_arr = np.random.uniform(-5, 5, size)
    z_arr = np.random.uniform(0.1, 3, size)  # Avoid zero for stability

    return x_arr, y_arr, z_arr


def validate_compatibility(x_arr, y_arr, z_arr, func_type='simple', tolerance=1e-10):
    """
    Validate that CPU and GPU implementations produce identical results.

    Args:
        x_arr, y_arr, z_arr: Input arrays
        func_type: Function type to test
        tolerance: Numerical tolerance for comparison

    Returns:
        Boolean indicating whether results match
    """
    # Compute on CPU
    cpu_results = cpu_compute_vectorized(x_arr, y_arr, z_arr, func_type)

    # Compute on GPU
    gpu_results = gpu_compute_wrapper(x_arr, y_arr, z_arr, func_type)

    # Compare results
    max_diff = np.max(np.abs(cpu_results - gpu_results))
    is_compatible = max_diff < tolerance

    return is_compatible, max_diff, cpu_results, gpu_results


def benchmark_performance(x_arr, y_arr, z_arr, func_type='simple', num_trials=5):
    """
    Benchmark CPU vs GPU performance for the compatible functions.

    Args:
        x_arr, y_arr, z_arr: Input arrays
        func_type: Function type to benchmark
        num_trials: Number of timing trials

    Returns:
        Dictionary with timing results
    """
    print(f"\n⚡ Performance Benchmark ({func_type} function):")
    print("-" * 50)

    # CPU timing
    cpu_times = []
    for _ in range(num_trials):
        start_time = time.time()
        cpu_results = cpu_compute_vectorized(x_arr, y_arr, z_arr, func_type)
        cpu_times.append((time.time() - start_time) * 1000)

    cpu_time_avg = np.mean(cpu_times)
    cpu_time_std = np.std(cpu_times)

    # GPU timing (including memory transfer)
    gpu_times = []
    for _ in range(num_trials):
        start_time = time.time()
        gpu_results = gpu_compute_wrapper(x_arr, y_arr, z_arr, func_type)
        gpu_times.append((time.time() - start_time) * 1000)

    gpu_time_avg = np.mean(gpu_times)
    gpu_time_std = np.std(gpu_times)

    # Results
    print(f"Data size: {len(x_arr):,} elements")
    print(f"CPU time: {cpu_time_avg:.3f} ± {cpu_time_std:.3f} ms")
    print(f"GPU time: {gpu_time_avg:.3f} ± {gpu_time_std:.3f} ms")

    if gpu_time_avg > 0:
        speedup = cpu_time_avg / gpu_time_avg
        print(f"Speedup: {speedup:.2f}x {'(GPU faster)' if speedup > 1 else '(CPU faster)'}")

    return {
        'cpu_time': cpu_time_avg,
        'gpu_time': gpu_time_avg,
        'speedup': cpu_time_avg / gpu_time_avg if gpu_time_avg > 0 else 0,
        'cpu_results': cpu_results,
        'gpu_results': gpu_results
    }


def demonstrate_compatibility():
    """
    Demonstrate the key concepts of CPU/GPU function compatibility.
    """
    print("🔄 CPU/GPU Function Compatibility Demonstration")
    print("=" * 60)

    # Original example from the file
    print("\n📋 Original Example:")
    print("-" * 20)

    # Single value computation
    result = business_logic(1, 2, 3)
    print(f"business_logic(1, 2, 3) = {result:.6f}")

    # Small array computation
    x_small = np.array([1, 10, 234], dtype=np.float64)
    y_small = np.array([2, 2, 4014], dtype=np.float64)
    z_small = np.array([3, 14, 2211], dtype=np.float64)

    print(f"\nInput arrays:")
    print(f"X = {x_small}")
    print(f"Y = {y_small}")
    print(f"Z = {z_small}")

    # CPU computation
    cpu_results_small = cpu_compute_vectorized(x_small, y_small, z_small, 'simple')
    print(f"\nCPU results: {cpu_results_small}")

    # GPU computation
    gpu_results_small = gpu_compute_wrapper(x_small, y_small, z_small, 'simple')
    print(f"GPU results: {gpu_results_small}")

    # Verify they match
    matches = np.allclose(cpu_results_small, gpu_results_small, rtol=1e-10)
    print(f"Results match: {'✅ Yes' if matches else '❌ No'}")


def main():
    """
    Main function demonstrating Numba JIT CPU/GPU compatibility.
    """
    print("🚀 Numba JIT CPU/GPU Compatibility Example")
    print("=" * 60)

    # Demonstrate basic compatibility
    demonstrate_compatibility()

    # Test with a medium-sized dataset
    print("\n🧪 Compatibility Testing:")
    print("-" * 30)

    try:
        # Generate test data
        x_arr, y_arr, z_arr = generate_test_data(1000)

        print(f"📊 Testing with {len(x_arr):,} elements:")

        # Test simple function
        is_compatible, max_diff, _, _ = validate_compatibility(
            x_arr, y_arr, z_arr, 'simple'
        )

        print(f"  Simple function compatibility: {'✅ Pass' if is_compatible else '❌ Fail'}")
        print(f"  Maximum difference: {max_diff:.2e}")

        # Test polynomial function
        is_compatible_poly, max_diff_poly, _, _ = validate_compatibility(
            x_arr, y_arr, z_arr, 'polynomial'
        )

        print(f"  Polynomial function compatibility: {'✅ Pass' if is_compatible_poly else '❌ Fail'}")
        print(f"  Maximum difference: {max_diff_poly:.2e}")

        # Performance benchmarking
        print("\n🏁 Performance Benchmarking:")
        print("-" * 30)

        # Benchmark simple function
        benchmark_performance(x_arr, y_arr, z_arr, 'simple')

        # Benchmark polynomial function
        benchmark_performance(x_arr, y_arr, z_arr, 'polynomial')

    except Exception as e:
        print(f"⚠️  GPU testing failed: {e}")
        print("Continuing with CPU-only demonstration...")

    # Educational summary
    print("\n🎓 Key Learning Points:")
    print("-" * 30)
    print("1. Functions decorated with @jit can be called from both CPU and GPU contexts")
    print("2. The same mathematical logic works identically on both platforms")
    print("3. GPU acceleration becomes beneficial for larger datasets")
    print("4. Memory transfer overhead affects GPU performance for small arrays")
    print("5. Simple mathematical operations are ideal for CPU/GPU compatibility")

    print("\n🔧 Technical Details:")
    print("-" * 20)
    print("• @jit functions compile to both CPU and GPU machine code")
    print("• Numba's unified type system ensures consistent behavior")
    print("• Mathematical operations maintain numerical precision")
    print("• Function calls from GPU kernels have zero overhead")
    print("• Same source code, multiple target architectures")

    print("\n✨ Use Cases:")
    print("-" * 15)
    print("• Scientific computing with mathematical models")
    print("• Financial calculations requiring high precision")
    print("• Image/signal processing with custom algorithms")
    print("• Machine learning with custom activation functions")
    print("• Physics simulations with shared computation kernels")


if __name__ == "__main__":
    main()
