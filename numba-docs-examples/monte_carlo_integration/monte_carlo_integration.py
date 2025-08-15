# import required libraries
import numpy as np
import time
import math
import warnings
from numba import jit, cuda
from numba.core.errors import NumbaPerformanceWarning
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float32,
)

# Suppress performance warnings for educational examples
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


@jit
def function_1_over_x(x):
    """Function: f(x) = 1/x"""
    return 1.0 / x


@jit
def function_x_squared(x):
    """Function: f(x) = x²"""
    return x * x


@jit
def function_sin(x):
    """Function: f(x) = sin(x)"""
    return math.sin(x)


@jit
def function_exp(x):
    """Function: f(x) = e^x"""
    return math.exp(x)


@jit
def function_polynomial(x):
    """Function: f(x) = x³ - 2x² + 3x - 1"""
    return x*x*x - 2*x*x + 3*x - 1


@cuda.jit
def mc_integrator_kernel(out, rng_states, lower_lim, upper_lim, func_selector):
    """
    CUDA kernel to draw random samples and evaluate the function.

    Args:
        out: Output array for function evaluations
        rng_states: Random number generator states
        lower_lim: Lower integration limit
        upper_lim: Upper integration limit
        func_selector: Integer selecting which function to integrate
    """
    gid = cuda.grid(1)

    if gid < len(out):
        # Generate random sample in [0, 1)
        samp = xoroshiro128p_uniform_float32(rng_states, gid)

        # Transform to integration interval [lower_lim, upper_lim]
        x = lower_lim + samp * (upper_lim - lower_lim)

        # Evaluate the selected function
        if func_selector == 0:
            y = function_1_over_x(x)
        elif func_selector == 1:
            y = function_x_squared(x)
        elif func_selector == 2:
            y = function_sin(x)
        elif func_selector == 3:
            y = function_exp(x)
        else:
            y = function_polynomial(x)

        out[gid] = y


@cuda.reduce
def sum_reduce(a, b):
    """CUDA reduction operation for summing array elements."""
    return a + b


def gpu_monte_carlo_integrate(lower_lim, upper_lim, nsamps, func_selector=0, block_size=512):
    """
    GPU-accelerated Monte Carlo integration.

    Args:
        lower_lim: Lower integration limit
        upper_lim: Upper integration limit
        nsamps: Number of random samples
        func_selector: Function to integrate (0-4)
        block_size: CUDA block size

    Returns:
        Approximated integral value
    """
    # Ensure good GPU occupancy
    min_samples = 32 * block_size  # At least 32 blocks
    if nsamps < min_samples:
        nsamps = min_samples

    # Allocate GPU memory
    out = cuda.device_array(nsamps, dtype=np.float32)
    rng_states = create_xoroshiro128p_states(nsamps, seed=42)

    # Calculate grid configuration
    num_blocks = (nsamps + block_size - 1) // block_size

    # Launch kernel
    mc_integrator_kernel[num_blocks, block_size](
        out, rng_states, lower_lim, upper_lim, func_selector
    )

    # Compute the integral using Monte Carlo formula
    # Integral ≈ (b - a) * (1/N) * Σf(xi)
    interval_width = upper_lim - lower_lim
    function_sum = sum_reduce(out)
    integral_estimate = interval_width * (function_sum / nsamps)

    return float(integral_estimate)


def cpu_monte_carlo_integrate(lower_lim, upper_lim, nsamps, func_selector=0):
    """
    CPU Monte Carlo integration for comparison.

    Args:
        lower_lim: Lower integration limit
        upper_lim: Upper integration limit
        nsamps: Number of random samples
        func_selector: Function to integrate (0-4)

    Returns:
        Approximated integral value
    """
    np.random.seed(42)  # For reproducible results

    # Generate random samples
    samples = np.random.uniform(lower_lim, upper_lim, nsamps)

    # Evaluate function at sample points
    if func_selector == 0:
        values = np.array([function_1_over_x(x) for x in samples])
    elif func_selector == 1:
        values = np.array([function_x_squared(x) for x in samples])
    elif func_selector == 2:
        values = np.array([function_sin(x) for x in samples])
    elif func_selector == 3:
        values = np.array([function_exp(x) for x in samples])
    else:
        values = np.array([function_polynomial(x) for x in samples])

    # Monte Carlo integration formula
    interval_width = upper_lim - lower_lim
    integral_estimate = interval_width * np.mean(values)

    return integral_estimate


def analytical_integral(lower_lim, upper_lim, func_selector=0):
    """
    Analytical solutions for comparison.

    Args:
        lower_lim: Lower integration limit
        upper_lim: Upper integration limit
        func_selector: Function to integrate (0-4)

    Returns:
        Exact integral value
    """
    if func_selector == 0:
        # ∫(1/x)dx = ln(x)
        return math.log(upper_lim) - math.log(lower_lim)
    elif func_selector == 1:
        # ∫(x²)dx = x³/3
        return (upper_lim**3 - lower_lim**3) / 3
    elif func_selector == 2:
        # ∫sin(x)dx = -cos(x)
        return -math.cos(upper_lim) + math.cos(lower_lim)
    elif func_selector == 3:
        # ∫e^x dx = e^x
        return math.exp(upper_lim) - math.exp(lower_lim)
    else:
        # ∫(x³ - 2x² + 3x - 1)dx = x⁴/4 - 2x³/3 + 3x²/2 - x
        def antiderivative(x):
            return x**4/4 - 2*x**3/3 + 3*x**2/2 - x
        return antiderivative(upper_lim) - antiderivative(lower_lim)


def benchmark_integration(lower_lim, upper_lim, nsamps, func_selector=0, num_trials=3):
    """
    Benchmark CPU vs GPU Monte Carlo integration.

    Args:
        lower_lim: Lower integration limit
        upper_lim: Upper integration limit
        nsamps: Number of samples
        func_selector: Function to integrate
        num_trials: Number of timing trials

    Returns:
        Dictionary with timing and accuracy results
    """
    function_names = ["1/x", "x²", "sin(x)", "e^x", "x³-2x²+3x-1"]
    func_name = function_names[func_selector]

    print(f"\n⚡ Benchmarking Integration of f(x) = {func_name}")
    print(f"   Interval: [{lower_lim}, {upper_lim}], Samples: {nsamps:,}")
    print("-" * 60)

    # Get analytical solution
    analytical_result = analytical_integral(lower_lim, upper_lim, func_selector)
    print(f"Analytical result: {analytical_result:.8f}")

    # CPU timing
    cpu_times = []
    for _ in range(num_trials):
        start_time = time.time()
        cpu_result = cpu_monte_carlo_integrate(lower_lim, upper_lim, nsamps, func_selector)
        cpu_times.append((time.time() - start_time) * 1000)

    cpu_time_avg = np.mean(cpu_times)
    cpu_error = abs(cpu_result - analytical_result)
    cpu_rel_error = cpu_error / abs(analytical_result) * 100

    # GPU timing
    gpu_times = []
    for _ in range(num_trials):
        start_time = time.time()
        gpu_result = gpu_monte_carlo_integrate(lower_lim, upper_lim, nsamps, func_selector)
        gpu_times.append((time.time() - start_time) * 1000)

    gpu_time_avg = np.mean(gpu_times)
    gpu_error = abs(gpu_result - analytical_result)
    gpu_rel_error = gpu_error / abs(analytical_result) * 100

    # Results
    print(f"CPU result:        {cpu_result:.8f} (error: {cpu_rel_error:.4f}%)")
    print(f"GPU result:        {gpu_result:.8f} (error: {gpu_rel_error:.4f}%)")
    print(f"CPU time:          {cpu_time_avg:.3f} ms")
    print(f"GPU time:          {gpu_time_avg:.3f} ms")

    if gpu_time_avg > 0:
        speedup = cpu_time_avg / gpu_time_avg
        print(f"Speedup:           {speedup:.2f}x {'(GPU faster)' if speedup > 1 else '(CPU faster)'}")

    return {
        'analytical': analytical_result,
        'cpu_result': cpu_result,
        'gpu_result': gpu_result,
        'cpu_time': cpu_time_avg,
        'gpu_time': gpu_time_avg,
        'cpu_error': cpu_rel_error,
        'gpu_error': gpu_rel_error,
        'speedup': cpu_time_avg / gpu_time_avg if gpu_time_avg > 0 else 0
    }


def convergence_analysis(lower_lim, upper_lim, func_selector=0):
    """
    Analyze convergence behavior with increasing sample sizes.
    """
    print(f"\n📈 Convergence Analysis")
    print("-" * 40)

    function_names = ["1/x", "x²", "sin(x)", "e^x", "x³-2x²+3x-1"]
    func_name = function_names[func_selector]
    analytical_result = analytical_integral(lower_lim, upper_lim, func_selector)

    print(f"Function: f(x) = {func_name}")
    print(f"Interval: [{lower_lim}, {upper_lim}]")
    print(f"Analytical: {analytical_result:.8f}")
    print()
    print(f"{'Samples':<12} {'GPU Result':<15} {'Error %':<10} {'Time (ms)':<10}")
    print("-" * 50)

    sample_sizes = [1000, 10000, 100000, 1000000]

    for nsamps in sample_sizes:
        start_time = time.time()
        gpu_result = gpu_monte_carlo_integrate(lower_lim, upper_lim, nsamps, func_selector)
        gpu_time = (time.time() - start_time) * 1000

        error_percent = abs(gpu_result - analytical_result) / abs(analytical_result) * 100

        print(f"{nsamps:<12,} {gpu_result:<15.8f} {error_percent:<10.4f} {gpu_time:<10.3f}")


def demonstrate_different_functions():
    """
    Demonstrate Monte Carlo integration on different mathematical functions.
    """
    print("🧮 Monte Carlo Integration Examples")
    print("=" * 60)

    # Test cases: (lower_lim, upper_lim, func_selector, description)
    test_cases = [
        (1, 2, 0, "∫₁² (1/x) dx = ln(2) ≈ 0.693147"),
        (0, 1, 1, "∫₀¹ x² dx = 1/3 ≈ 0.333333"),
        (0, math.pi/2, 2, "∫₀^(π/2) sin(x) dx = 1"),
        (0, 1, 3, "∫₀¹ eˣ dx = e - 1 ≈ 1.718282"),
        (0, 2, 4, "∫₀² (x³-2x²+3x-1) dx = 8/3 ≈ 2.666667")
    ]

    for lower, upper, func_sel, description in test_cases:
        print(f"\n📊 Example: {description}")
        benchmark_integration(lower, upper, 100000, func_sel, num_trials=1)


def main():
    """
    Main function demonstrating Monte Carlo integration with CUDA.
    """
    print("🎲 CUDA Monte Carlo Integration")
    print("=" * 50)

    try:
        # Original examples from the file
        print("\n📋 Original Examples:")
        print("-" * 25)

        # Test case 1: ∫₁² (1/x) dx
        result1 = gpu_monte_carlo_integrate(1, 2, 1000000, 0)
        analytical1 = analytical_integral(1, 2, 0)
        error1 = abs(result1 - analytical1) / analytical1 * 100
        print(f"∫₁² (1/x) dx:")
        print(f"  GPU result:    {result1:.6f}")
        print(f"  Analytical:    {analytical1:.6f}")
        print(f"  Error:         {error1:.4f}%")

        # Test case 2: ∫₂³ (1/x) dx
        result2 = gpu_monte_carlo_integrate(2, 3, 1000000, 0)
        analytical2 = analytical_integral(2, 3, 0)
        error2 = abs(result2 - analytical2) / analytical2 * 100
        print(f"\n∫₂³ (1/x) dx:")
        print(f"  GPU result:    {result2:.6f}")
        print(f"  Analytical:    {analytical2:.6f}")
        print(f"  Error:         {error2:.4f}%")

        # Demonstrate different functions
        demonstrate_different_functions()

        # Convergence analysis
        convergence_analysis(1, 2, 0)  # Analyze ln(2) convergence

        print("\n🎓 Key Learning Points:")
        print("-" * 30)
        print("1. Monte Carlo integration approximates integrals using random sampling")
        print("2. Accuracy improves with more samples (√N convergence rate)")
        print("3. GPU acceleration becomes beneficial for large sample sizes")
        print("4. Different functions have different convergence characteristics")
        print("5. Random number generation is crucial for Monte Carlo methods")

        print("\n🔧 Technical Details:")
        print("-" * 20)
        print("• Uses Xoroshiro128+ PRNG for high-quality random numbers")
        print("• CUDA reduction for efficient parallel summation")
        print("• Multiple function support through kernel selection")
        print("• Proper Monte Carlo formula: (b-a) * (1/N) * Σf(xi)")
        print("• Optimized GPU occupancy with minimum block requirements")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("Make sure you have a CUDA-capable GPU and proper CUDA installation.")


if __name__ == "__main__":
    main()
