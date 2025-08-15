#!/usr/bin/env python3
"""
Interactive CUDA Matrix Multiplication with User Input

This program allows users to specify custom matrix dimensions and compare
different matrix multiplication implementations (CPU, GPU naive, GPU optimized).
Demonstrates scaling behavior and performance characteristics across different
problem sizes with real-time user interaction.
"""

import numpy as np
import time
import warnings
import sys
from numba import cuda, float32, jit
from numba.core.errors import NumbaPerformanceWarning

# Suppress performance warnings for cleaner output
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

# Optimal tile size for shared memory
TILE_SIZE = 16


@cuda.jit
def naive_matmul(A, B, C):
    """Naive matrix multiplication kernel using global memory only."""
    i, j = cuda.grid(2)
    
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


@cuda.jit
def shared_memory_matmul(A, B, C):
    """Optimized matrix multiplication using shared memory tiling."""
    # Shared memory arrays for tiles
    sA = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)
    sB = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=float32)
    
    # Thread and block indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    
    # Global thread indices
    row = by * TILE_SIZE + ty
    col = bx * TILE_SIZE + tx
    
    # Accumulator for the result
    tmp = 0.0
    
    # Number of tiles needed to cover the K dimension
    num_tiles = (A.shape[1] + TILE_SIZE - 1) // TILE_SIZE
    
    # Loop over tiles
    for tile in range(num_tiles):
        # Load tile into shared memory
        if row < A.shape[0] and (tile * TILE_SIZE + tx) < A.shape[1]:
            sA[ty, tx] = A[row, tile * TILE_SIZE + tx]
        else:
            sA[ty, tx] = 0.0
            
        if (tile * TILE_SIZE + ty) < B.shape[0] and col < B.shape[1]:
            sB[ty, tx] = B[tile * TILE_SIZE + ty, col]
        else:
            sB[ty, tx] = 0.0
        
        # Synchronize to ensure all threads have loaded their data
        cuda.syncthreads()
        
        # Compute partial dot product using shared memory
        for k in range(TILE_SIZE):
            tmp += sA[ty, k] * sB[k, tx]
        
        # Synchronize before loading next tile
        cuda.syncthreads()
    
    # Write result to global memory
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = tmp


@jit(nopython=True)
def cpu_matmul(A, B):
    """CPU matrix multiplication with Numba JIT optimization."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Matrix dimensions don't match"
    
    C = np.zeros((M, N), dtype=np.float32)
    
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
    
    return C


def gpu_matmul_naive(A, B, block_size=(16, 16)):
    """GPU matrix multiplication using naive algorithm."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Matrix dimensions don't match: {K} != {K2}"
    
    # Allocate GPU memory
    d_A = cuda.to_device(A.astype(np.float32))
    d_B = cuda.to_device(B.astype(np.float32))
    d_C = cuda.device_array((M, N), dtype=np.float32)
    
    # Calculate grid dimensions
    blocks_per_grid_x = (N + block_size[0] - 1) // block_size[0]
    blocks_per_grid_y = (M + block_size[1] - 1) // block_size[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Launch kernel
    naive_matmul[blocks_per_grid, block_size](d_A, d_B, d_C)
    
    # Copy result back to host
    return d_C.copy_to_host()


def gpu_matmul_shared(A, B, tile_size=TILE_SIZE):
    """GPU matrix multiplication using shared memory optimization."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Matrix dimensions don't match: {K} != {K2}"
    
    # Allocate GPU memory
    d_A = cuda.to_device(A.astype(np.float32))
    d_B = cuda.to_device(B.astype(np.float32))
    d_C = cuda.device_array((M, N), dtype=np.float32)
    
    # Calculate grid dimensions
    block_size = (tile_size, tile_size)
    blocks_per_grid_x = (N + tile_size - 1) // tile_size
    blocks_per_grid_y = (M + tile_size - 1) // tile_size
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Launch kernel
    shared_memory_matmul[blocks_per_grid, block_size](d_A, d_B, d_C)
    
    # Copy result back to host
    return d_C.copy_to_host()


def generate_test_matrices(M, N, K, matrix_type='random', seed=42):
    """Generate test matrices with specified type and dimensions."""
    np.random.seed(seed)
    
    if matrix_type == 'random':
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
    elif matrix_type == 'sequential':
        A = np.arange(M * K).reshape(M, K).astype(np.float32)
        B = np.arange(K * N).reshape(K, N).astype(np.float32)
    elif matrix_type == 'identity':
        A = np.eye(M, K).astype(np.float32)
        B = np.eye(K, N).astype(np.float32)
    else:  # ones
        A = np.ones((M, K), dtype=np.float32)
        B = np.ones((K, N), dtype=np.float32)
    
    return A, B


def estimate_memory_usage(M, N, K):
    """Estimate GPU memory usage for given matrix dimensions."""
    # Memory for matrices A, B, C in bytes (float32 = 4 bytes)
    memory_bytes = (M * K + K * N + M * N) * 4
    memory_mb = memory_bytes / (1024 * 1024)
    return memory_mb


def validate_results(C_cpu, C_gpu_naive, C_gpu_shared, tolerance=1e-4):
    """Validate that all implementations produce consistent results."""
    naive_match = np.allclose(C_cpu, C_gpu_naive, rtol=tolerance, atol=tolerance)
    shared_match = np.allclose(C_cpu, C_gpu_shared, rtol=tolerance, atol=tolerance)
    
    return naive_match and shared_match, naive_match, shared_match


def get_user_input():
    """Get matrix dimensions and preferences from user."""
    print("\n🔢 Interactive CUDA Matrix Multiplication")
    print("=" * 60)
    print("This program will compute C = A × B where:")
    print("  • A is an M×K matrix")
    print("  • B is a K×N matrix") 
    print("  • C is the resulting M×N matrix")
    print()
    
    try:
        # Get matrix dimensions
        M = int(input("Enter number of rows for matrix A (M): "))
        K = int(input("Enter number of columns for A / rows for B (K): "))
        N = int(input("Enter number of columns for matrix B (N): "))
        
        # Validate dimensions
        if M <= 0 or K <= 0 or N <= 0:
            raise ValueError("All dimensions must be positive integers")
        
        # Estimate memory usage
        memory_mb = estimate_memory_usage(M, N, K)
        print(f"\nEstimated GPU memory usage: {memory_mb:.1f} MB")
        
        if memory_mb > 1000:  # Warn for large matrices
            response = input("⚠️  Large matrix detected. Continue? (y/n): ")
            if response.lower() != 'y':
                return None, None, None, None, None
        
        # Get matrix type
        print("\nMatrix generation options:")
        print("1. Random values (default)")
        print("2. Sequential values (0, 1, 2, ...)")
        print("3. Identity matrices")
        print("4. All ones")
        
        choice = input("Choose matrix type (1-4, default=1): ").strip()
        matrix_types = {'1': 'random', '2': 'sequential', '3': 'identity', '4': 'ones'}
        matrix_type = matrix_types.get(choice, 'random')
        
        # Get number of trials
        trials = input("Number of timing trials (default=3): ").strip()
        num_trials = int(trials) if trials.isdigit() else 3
        
        return M, N, K, matrix_type, num_trials
        
    except (ValueError, KeyboardInterrupt) as e:
        print(f"Invalid input: {e}")
        return None, None, None, None, None


def benchmark_user_matrices(M, N, K, matrix_type, num_trials):
    """Benchmark matrix multiplication with user-specified dimensions."""
    print(f"\n🚀 Running Benchmark")
    print(f"Dimensions: A({M}×{K}) × B({K}×{N}) = C({M}×{N})")
    print(f"Matrix type: {matrix_type}")
    print(f"Trials: {num_trials}")
    print("-" * 60)
    
    # Generate test matrices
    print("Generating matrices...")
    A, B = generate_test_matrices(M, N, K, matrix_type)
    
    results = {}
    
    # CPU benchmark
    print("Running CPU benchmark...")
    cpu_times = []
    for trial in range(num_trials):
        print(f"  CPU trial {trial + 1}/{num_trials}", end='\r')
        start_time = time.time()
        C_cpu = cpu_matmul(A, B)
        cpu_times.append((time.time() - start_time) * 1000)
    
    results['cpu_time'] = np.mean(cpu_times)
    print(f"  CPU average: {results['cpu_time']:.3f} ms")
    
    # GPU naive benchmark
    print("Running GPU naive benchmark...")
    gpu_naive_times = []
    for trial in range(num_trials):
        print(f"  GPU naive trial {trial + 1}/{num_trials}", end='\r')
        start_time = time.time()
        C_gpu_naive = gpu_matmul_naive(A, B)
        gpu_naive_times.append((time.time() - start_time) * 1000)
    
    results['gpu_naive_time'] = np.mean(gpu_naive_times)
    print(f"  GPU naive average: {results['gpu_naive_time']:.3f} ms")
    
    # GPU shared memory benchmark
    print("Running GPU shared memory benchmark...")
    gpu_shared_times = []
    for trial in range(num_trials):
        print(f"  GPU shared trial {trial + 1}/{num_trials}", end='\r')
        start_time = time.time()
        C_gpu_shared = gpu_matmul_shared(A, B)
        gpu_shared_times.append((time.time() - start_time) * 1000)
    
    results['gpu_shared_time'] = np.mean(gpu_shared_times)
    print(f"  GPU shared average: {results['gpu_shared_time']:.3f} ms")
    
    # Validate results
    print("\nValidating results...")
    is_valid, naive_valid, shared_valid = validate_results(C_cpu, C_gpu_naive, C_gpu_shared)
    
    # Calculate speedups
    results['naive_speedup'] = results['cpu_time'] / results['gpu_naive_time'] if results['gpu_naive_time'] > 0 else 0
    results['shared_speedup'] = results['cpu_time'] / results['gpu_shared_time'] if results['gpu_shared_time'] > 0 else 0
    results['shared_vs_naive'] = results['gpu_naive_time'] / results['gpu_shared_time'] if results['gpu_shared_time'] > 0 else 0
    
    # Display results with proper alignment
    print("\n📊 BENCHMARK RESULTS")
    print("=" * 80)

    # Calculate GFLOPS
    total_ops = 2 * M * N * K  # Multiply-add operations
    cpu_gflops = total_ops / (results['cpu_time'] * 1e6)
    naive_gflops = total_ops / (results['gpu_naive_time'] * 1e6)
    shared_gflops = total_ops / (results['gpu_shared_time'] * 1e6)

    # Main results table with fixed-width formatting
    print(f"{'Method':<22} {'Time (ms)':>10} {'Speedup':>10} {'GFLOPS':>10}")
    print("-" * 80)
    print(f"{'CPU (Numba JIT)':<22} {results['cpu_time']:>10.3f} {'1.00x':>10} {cpu_gflops:>10.2f}")
    print(f"{'GPU (Naive)':<22} {results['gpu_naive_time']:>10.3f} {results['naive_speedup']:>9.2f}x {naive_gflops:>10.2f}")
    print(f"{'GPU (Shared Memory)':<22} {results['gpu_shared_time']:>10.3f} {results['shared_speedup']:>9.2f}x {shared_gflops:>10.2f}")

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("-" * 80)
    print(f"{'Metric':<25} {'Value':>15}")
    print("-" * 45)
    print(f"{'Shared vs Naive':<25} {results['shared_vs_naive']:>14.2f}x")
    print(f"{'Best GPU vs CPU':<25} {max(results['naive_speedup'], results['shared_speedup']):>14.2f}x")
    best_method = "Shared Memory" if results['shared_speedup'] > results['naive_speedup'] else "Naive"
    print(f"{'Best GPU Method':<25} {best_method:>15}")
    print(f"{'Peak GFLOPS':<25} {max(naive_gflops, shared_gflops):>14.2f}")

    print(f"\n🔍 Validation Results:")
    print(f"  Overall:             {'✅ All match' if is_valid else '❌ Mismatch detected'}")
    if not is_valid:
        print(f"  Naive vs CPU:        {'✅ Match' if naive_valid else '❌ Mismatch'}")
        print(f"  Shared vs CPU:       {'✅ Match' if shared_valid else '❌ Mismatch'}")

    # Performance analysis
    print(f"\n🧠 Performance Analysis")
    print("-" * 30)
    print(f"Matrix dimensions:     {M} × {K} × {N}")
    print(f"Total operations:      {total_ops:,}")
    print(f"Memory usage (est):    {estimate_memory_usage(M, N, K):.1f} MB")
    
    return results


def quick_benchmark_mode():
    """Quick benchmark mode with predefined sizes."""
    print("\n🚀 Quick Benchmark Mode")
    print("Testing predefined matrix sizes...")

    test_sizes = [
        (64, 64, 64, "Small matrices"),
        (128, 128, 128, "Medium matrices"),
        (256, 256, 256, "Large matrices"),
        (512, 512, 512, "Very large matrices")
    ]

    print(f"\n{'Matrix Size':<12} {'Description':<18} {'CPU (ms)':>10} {'GPU Naive':>12} {'GPU Shared':>12} {'Best Speedup':>12}")
    print("-" * 85)

    for M, N, K, description in test_sizes:
        try:
            A, B = generate_test_matrices(M, N, K, 'random')

            # Quick single-trial benchmark
            start = time.time()
            cpu_matmul(A, B)  # We don't need to store the result for timing
            cpu_time = (time.time() - start) * 1000

            start = time.time()
            gpu_matmul_naive(A, B)
            gpu_naive_time = (time.time() - start) * 1000

            start = time.time()
            gpu_matmul_shared(A, B)
            gpu_shared_time = (time.time() - start) * 1000

            # Calculate best speedup
            best_speedup = max(cpu_time / gpu_naive_time, cpu_time / gpu_shared_time)

            matrix_size = f"{M}×{N}×{K}"
            print(f"{matrix_size:<12} {description:<18} {cpu_time:>10.1f} {gpu_naive_time:>12.1f} {gpu_shared_time:>12.1f} {best_speedup:>11.2f}x")

        except Exception as e:
            matrix_size = f"{M}×{N}×{K}"
            print(f"{matrix_size:<12} {description:<18} {'Error':>10} {str(e)[:40]:>12}")
            break


def show_help():
    """Display help information."""
    print("\n📚 CUDA Matrix Multiplication Help")
    print("=" * 50)
    print("This program demonstrates GPU-accelerated matrix multiplication")
    print("using CUDA and compares different implementation strategies.")
    print()
    print("🔧 Implementation Details:")
    print("• CPU: Numba JIT-compiled triple nested loop")
    print("• GPU Naive: Global memory only, one thread per result element")
    print("• GPU Shared: Tiled algorithm using shared memory optimization")
    print()
    print("📊 Performance Factors:")
    print("• Small matrices: CPU often faster due to GPU launch overhead")
    print("• Large matrices: GPU shows significant speedup")
    print("• Shared memory: Reduces global memory traffic")
    print("• Memory coalescing: Consecutive threads access consecutive data")
    print()
    print("💡 Tips:")
    print("• Try different matrix sizes to see scaling behavior")
    print("• Square matrices (M=N=K) are most common in practice")
    print("• Large matrices may require significant GPU memory")
    print("• Multiple trials provide more accurate timing measurements")


def main():
    """Main interactive function with enhanced features."""
    print("🔢 Interactive CUDA Matrix Multiplication")
    print("=" * 60)
    print("Choose an option:")
    print("1. Custom matrix dimensions (interactive)")
    print("2. Quick benchmark (predefined sizes)")
    print("3. Help and information")
    print("4. Exit")

    try:
        while True:
            print(f"\n{'='*60}")
            choice = input("Enter your choice (1-4): ").strip()

            if choice == '1':
                # Interactive mode
                while True:
                    M, N, K, matrix_type, num_trials = get_user_input()

                    if M is None:  # User cancelled or invalid input
                        break

                    # Run benchmark
                    benchmark_user_matrices(M, N, K, matrix_type, num_trials)

                    # Ask if user wants to continue with custom dimensions
                    response = input("\nRun another custom benchmark? (y/n): ")
                    if response.lower() != 'y':
                        break

            elif choice == '2':
                # Quick benchmark mode
                quick_benchmark_mode()

            elif choice == '3':
                # Help
                show_help()

            elif choice == '4':
                # Exit
                break

            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                continue

        print("\n👋 Thank you for using Interactive CUDA Matrix Multiplication!")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have a CUDA-capable GPU and proper CUDA installation.")


if __name__ == "__main__":
    main()
