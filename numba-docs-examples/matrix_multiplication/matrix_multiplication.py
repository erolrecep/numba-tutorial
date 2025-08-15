# import required libraries
import numpy as np
import time
import warnings
from numba import cuda, float32, jit
from numba.core.errors import NumbaPerformanceWarning

# Suppress performance warnings for educational examples
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

# Tile size for shared memory optimization
TILE_SIZE = 16


@cuda.jit
def naive_matmul(A, B, C):
    """
    Naive matrix multiplication kernel: C = A * B

    Each thread computes one element of the result matrix.
    Uses global memory for all accesses - not optimized.

    Time Complexity: O(N³) operations
    Memory Complexity: O(N²) global memory accesses per element
    """
    i, j = cuda.grid(2)

    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


@cuda.jit
def shared_memory_matmul(A, B, C):
    """
    Optimized matrix multiplication using shared memory tiling.

    Divides matrices into tiles that fit in shared memory,
    reducing global memory accesses and improving performance.

    Reference: CUDA Programming Guide, Matrix Multiplication Example
    """
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
        # Each thread loads one element
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
    """
    CPU matrix multiplication for comparison.
    Uses Numba JIT for fair performance comparison.
    """
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
    """
    GPU matrix multiplication using naive algorithm.

    Args:
        A, B: Input matrices (NumPy arrays)
        block_size: CUDA block dimensions

    Returns:
        Result matrix C = A @ B
    """
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
    """
    GPU matrix multiplication using shared memory optimization.

    Args:
        A, B: Input matrices (NumPy arrays)
        tile_size: Size of shared memory tiles

    Returns:
        Result matrix C = A @ B
    """
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


def generate_test_matrices(M, N, K, matrix_type='random'):
    """
    Generate test matrices for benchmarking.

    Args:
        M, N, K: Matrix dimensions (A: M×K, B: K×N, C: M×N)
        matrix_type: Type of matrices to generate

    Returns:
        Tuple of (A, B) matrices
    """
    if matrix_type == 'random':
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
    elif matrix_type == 'sequential':
        A = np.arange(M * K).reshape(M, K).astype(np.float32)
        B = np.arange(K * N).reshape(K, N).astype(np.float32)
    elif matrix_type == 'identity':
        A = np.eye(M, K).astype(np.float32)
        B = np.eye(K, N).astype(np.float32)
    else:
        # Default to ones
        A = np.ones((M, K), dtype=np.float32)
        B = np.ones((K, N), dtype=np.float32)

    return A, B


def validate_results(C_cpu, C_gpu_naive, C_gpu_shared, tolerance=1e-4):
    """
    Validate that all implementations produce the same results.

    Args:
        C_cpu, C_gpu_naive, C_gpu_shared: Result matrices
        tolerance: Numerical tolerance for comparison

    Returns:
        Boolean indicating if all results match
    """
    naive_match = np.allclose(C_cpu, C_gpu_naive, rtol=tolerance, atol=tolerance)
    shared_match = np.allclose(C_cpu, C_gpu_shared, rtol=tolerance, atol=tolerance)

    return naive_match and shared_match, naive_match, shared_match


def benchmark_matrix_multiplication(M, N, K, num_trials=3):
    """
    Benchmark different matrix multiplication implementations.

    Args:
        M, N, K: Matrix dimensions
        num_trials: Number of timing trials

    Returns:
        Dictionary with timing results
    """
    print(f"\n🔢 Matrix Multiplication Benchmark")
    print(f"   Dimensions: A({M}×{K}) × B({K}×{N}) = C({M}×{N})")
    print("-" * 60)

    # Generate test matrices
    A, B = generate_test_matrices(M, N, K, 'random')

    # CPU timing
    cpu_times = []
    for _ in range(num_trials):
        start_time = time.time()
        C_cpu = cpu_matmul(A, B)
        cpu_times.append((time.time() - start_time) * 1000)

    cpu_time_avg = np.mean(cpu_times)

    # GPU naive timing
    gpu_naive_times = []
    for _ in range(num_trials):
        start_time = time.time()
        C_gpu_naive = gpu_matmul_naive(A, B)
        gpu_naive_times.append((time.time() - start_time) * 1000)

    gpu_naive_time_avg = np.mean(gpu_naive_times)

    # GPU shared memory timing
    gpu_shared_times = []
    for _ in range(num_trials):
        start_time = time.time()
        C_gpu_shared = gpu_matmul_shared(A, B)
        gpu_shared_times.append((time.time() - start_time) * 1000)

    gpu_shared_time_avg = np.mean(gpu_shared_times)

    # Validate results
    is_valid, naive_valid, shared_valid = validate_results(C_cpu, C_gpu_naive, C_gpu_shared)

    # Results
    print(f"CPU (Numba JIT):       {cpu_time_avg:.3f} ms")
    print(f"GPU (Naive):           {gpu_naive_time_avg:.3f} ms")
    print(f"GPU (Shared Memory):   {gpu_shared_time_avg:.3f} ms")

    if gpu_naive_time_avg > 0:
        naive_speedup = cpu_time_avg / gpu_naive_time_avg
        print(f"Naive GPU Speedup:     {naive_speedup:.2f}x")

    if gpu_shared_time_avg > 0:
        shared_speedup = cpu_time_avg / gpu_shared_time_avg
        shared_vs_naive = gpu_naive_time_avg / gpu_shared_time_avg
        print(f"Shared GPU Speedup:    {shared_speedup:.2f}x")
        print(f"Shared vs Naive:       {shared_vs_naive:.2f}x")

    print(f"Results Valid:         {'✅ All match' if is_valid else '❌ Mismatch detected'}")
    if not is_valid:
        print(f"  Naive vs CPU:        {'✅' if naive_valid else '❌'}")
        print(f"  Shared vs CPU:       {'✅' if shared_valid else '❌'}")

    return {
        'cpu_time': cpu_time_avg,
        'gpu_naive_time': gpu_naive_time_avg,
        'gpu_shared_time': gpu_shared_time_avg,
        'naive_speedup': cpu_time_avg / gpu_naive_time_avg if gpu_naive_time_avg > 0 else 0,
        'shared_speedup': cpu_time_avg / gpu_shared_time_avg if gpu_shared_time_avg > 0 else 0,
        'is_valid': is_valid
    }


def demonstrate_original_example():
    """
    Demonstrate the original example with improved implementation.
    """
    print("📋 Original Example (4×4 matrices):")
    print("-" * 40)

    # Original matrices from the file
    A = np.arange(16).reshape([4, 4]).astype(np.float32)
    B = np.ones([4, 4], dtype=np.float32)

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)

    # CPU reference
    C_cpu = cpu_matmul(A, B)
    print(f"\nCPU Result (A @ B):")
    print(C_cpu)

    # GPU naive
    C_gpu_naive = gpu_matmul_naive(A, B)
    print(f"\nGPU Naive Result:")
    print(C_gpu_naive)

    # GPU shared memory
    C_gpu_shared = gpu_matmul_shared(A, B)
    print(f"\nGPU Shared Memory Result:")
    print(C_gpu_shared)

    # Validation
    is_valid, _, _ = validate_results(C_cpu, C_gpu_naive, C_gpu_shared)
    print(f"\nResults match: {'✅ Yes' if is_valid else '❌ No'}")


def scalability_analysis():
    """
    Analyze performance scaling with different matrix sizes.
    """
    print("\n📈 Scalability Analysis")
    print("=" * 50)

    # Test different matrix sizes
    sizes = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024)
    ]

    print(f"{'Size':<12} {'CPU (ms)':<12} {'Naive (ms)':<12} {'Shared (ms)':<12} {'Best Speedup':<12}")
    print("-" * 65)

    for M, N, K in sizes:
        try:
            results = benchmark_matrix_multiplication(M, N, K, num_trials=1)
            best_speedup = max(results['naive_speedup'], results['shared_speedup'])

            print(f"{M}×{N}×{K:<6} {results['cpu_time']:<12.1f} {results['gpu_naive_time']:<12.1f} "
                  f"{results['gpu_shared_time']:<12.1f} {best_speedup:<12.2f}x")

        except Exception as e:
            print(f"{M}×{N}×{K:<6} Error: {str(e)[:40]}...")
            break


def memory_analysis():
    """
    Analyze memory usage and shared memory benefits.
    """
    print("\n🧠 Memory Access Analysis")
    print("-" * 40)

    M, N, K = 256, 256, 256

    print(f"Matrix dimensions: {M}×{N}×{K}")
    print(f"Total elements: {M*N + N*K + M*K:,}")
    print(f"Memory usage: {(M*N + N*K + M*K) * 4 / 1024:.1f} KB")

    print(f"\nNaive Algorithm:")
    print(f"  Global memory accesses per element: {K} reads + 1 write = {K+1}")
    print(f"  Total global memory accesses: {M*N*(K+1):,}")

    print(f"\nShared Memory Algorithm:")
    tile_size = TILE_SIZE
    tiles_per_dim = (max(M, N) + tile_size - 1) // tile_size
    print(f"  Tile size: {tile_size}×{tile_size}")
    print(f"  Tiles per dimension: ~{tiles_per_dim}")
    print(f"  Shared memory per block: {2 * tile_size * tile_size * 4} bytes")
    print(f"  Reduced global memory traffic through data reuse")


def main():
    """
    Main function demonstrating CUDA matrix multiplication.
    """
    print("🔢 CUDA Matrix Multiplication")
    print("=" * 50)

    try:
        # Demonstrate original example
        demonstrate_original_example()

        # Benchmark different sizes
        print("\n🏁 Performance Benchmarks:")
        print("=" * 30)

        # Small matrices
        benchmark_matrix_multiplication(64, 64, 64)

        # Medium matrices
        benchmark_matrix_multiplication(256, 256, 256)

        # Large matrices (if GPU memory allows)
        try:
            benchmark_matrix_multiplication(512, 512, 512)
        except Exception as e:
            print(f"Large matrix benchmark failed: {e}")

        # Memory analysis
        memory_analysis()

        # Scalability analysis
        scalability_analysis()

        print("\n🎓 Key Learning Points:")
        print("-" * 30)
        print("1. Shared memory reduces global memory traffic")
        print("2. Tiling strategy improves cache locality")
        print("3. GPU advantage increases with matrix size")
        print("4. Memory coalescing is crucial for performance")
        print("5. Thread block size affects occupancy")

        print("\n🔧 Technical Details:")
        print("-" * 20)
        print(f"• Tile size: {TILE_SIZE}×{TILE_SIZE}")
        print("• Shared memory usage: 2 tiles per thread block")
        print("• Memory coalescing: Consecutive threads access consecutive elements")
        print("• Synchronization: cuda.syncthreads() ensures data consistency")
        print("• Bounds checking: Handles non-square and non-tile-aligned matrices")

        print("\n✨ Applications:")
        print("-" * 15)
        print("• Deep learning: Neural network training and inference")
        print("• Scientific computing: Linear algebra operations")
        print("• Computer graphics: 3D transformations and rendering")
        print("• Signal processing: Convolution and filtering")
        print("• Numerical simulations: Finite element methods")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("Make sure you have a CUDA-capable GPU and proper CUDA installation.")


if __name__ == "__main__":
    main()
