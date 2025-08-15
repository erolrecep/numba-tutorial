# CUDA Vector Addition with Numba

This example demonstrates GPU-accelerated vector addition using Numba's CUDA JIT compiler. It showcases parallel computing concepts, performance comparison between CPU and GPU implementations, and two different methods for launching CUDA kernels.

## Overview

Vector addition is a fundamental parallel computing operation where two vectors of equal length are added element-wise to produce a result vector: `C = A + B`. This operation is embarrassingly parallel, making it ideal for GPU acceleration since each element can be computed independently.

## Technical Implementation

### CUDA Kernel Design

```python
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
```

#### Key Technical Concepts:

1. **Thread Indexing**: `cuda.grid(1)` calculates the global thread ID in a 1D grid
   - Formula: `threadIdx.x + (blockIdx.x * blockDim.x)`
   - Each thread gets a unique ID to process one vector element

2. **Bounds Checking**: `if tid < size:` prevents out-of-bounds memory access
   - Essential when total threads > vector size
   - Prevents undefined behavior and potential crashes

3. **Memory Access Pattern**: Coalesced memory access
   - Consecutive threads access consecutive memory locations
   - Maximizes memory bandwidth utilization

### Memory Management

The implementation demonstrates proper CUDA memory management:

```python
# Transfer data to GPU device memory
a_gpu = cuda.to_device(a_cpu)      # Host → Device transfer
b_gpu = cuda.to_device(b_cpu)      # Host → Device transfer
c_gpu = cuda.device_array_like(a_gpu)  # Allocate device memory

# Compute on GPU
vector_add_gpu[nblocks, nthreads](a_gpu, b_gpu, c_gpu)

# Transfer result back to host
c_result = c_gpu.copy_to_host()    # Device → Host transfer
```

#### Memory Transfer Overhead:
- Host-to-device and device-to-host transfers are expensive
- For small vectors, transfer time may exceed computation time
- GPU acceleration becomes beneficial for larger datasets

### Kernel Launch Methods

The example demonstrates two kernel launch approaches:

#### Method 1: Automatic Configuration (`forall`)
```python
vector_add_gpu.forall(len(a_gpu))(a_gpu, b_gpu, c_gpu)
```
- **Advantages**: Numba automatically determines optimal block/thread configuration
- **Use Case**: Quick prototyping, when you don't need fine-grained control
- **Performance**: Generally good, but may not be optimal for specific hardware

#### Method 2: Manual Configuration
```python
nthreads = 256  # Threads per block
nblocks = (N + nthreads - 1) // nthreads  # Ceiling division
vector_add_gpu[nblocks, nthreads](a_gpu, b_gpu, c_gpu)
```
- **Thread Block Size**: 256 threads per block
  - Multiple of warp size (32) for efficient execution
  - Allows several warps per block for better occupancy
- **Block Calculation**: Ceiling division ensures all elements are covered
  - `(N + nthreads - 1) // nthreads` avoids integer truncation
- **Advantages**: Fine-grained control, potential for optimization

### Performance Considerations

#### GPU Architecture Factors:
1. **Warp Size**: NVIDIA GPUs execute threads in groups of 32 (warps)
2. **Occupancy**: Ratio of active warps to maximum possible warps
3. **Memory Bandwidth**: GPU memory bandwidth >> CPU memory bandwidth
4. **Latency Hiding**: GPU hides memory latency with massive parallelism

#### Performance Metrics:
- **Speedup**: Ratio of CPU time to GPU time
- **Throughput**: Elements processed per second
- **Memory Bandwidth Utilization**: Actual vs. theoretical bandwidth

## Code Structure

### Main Components:

1. **`vector_add_gpu()`**: CUDA kernel function
2. **`vector_add_cpu()`**: CPU reference implementation
3. **`main()`**: Orchestrates execution and benchmarking

### Execution Flow:

1. **Data Generation**: Create random input vectors on CPU
2. **CPU Baseline**: Compute reference result and timing
3. **GPU Method 1**: Execute with automatic configuration
4. **GPU Method 2**: Execute with manual configuration
5. **Verification**: Validate GPU results against CPU reference
6. **Performance Analysis**: Compare execution times and speedups

## Requirements

- **Hardware**: CUDA-capable NVIDIA GPU
- **Software**:
  - Python 3.6+
  - NumPy
  - Numba with CUDA support
  - CUDA Toolkit

## Usage

```bash
python vector_addition.py
```

### Expected Output:
```
Vector size: 100,000

--- CPU Computation ---
CPU time: 0.001234 seconds

--- GPU Computation ---
Method 1: Using forall (automatic configuration)
GPU time (forall): 0.000567 seconds

Method 2: Manual block/thread configuration
Blocks: 391, Threads per block: 256
Total threads: 100,096
GPU time (manual): 0.000543 seconds

--- Verification ---
Forall method correct: True
Manual method correct: True

--- Performance Summary ---
CPU time:           0.001234 seconds
GPU time (forall):  0.000567 seconds
GPU time (manual):  0.000543 seconds
Speedup (forall):   2.18x
Speedup (manual):   2.27x
```

## Learning Objectives

This example teaches:

1. **CUDA Programming Fundamentals**:
   - Thread indexing and grid organization
   - Memory management between host and device
   - Kernel launch configurations

2. **Parallel Algorithm Design**:
   - Identifying embarrassingly parallel problems
   - Thread-to-data mapping strategies
   - Bounds checking and edge cases

3. **Performance Analysis**:
   - CPU vs GPU performance comparison
   - Understanding when GPU acceleration is beneficial
   - Memory transfer overhead considerations

4. **Numba CUDA Features**:
   - `@cuda.jit` decorator usage
   - Different kernel launch methods
   - Device memory management functions

## Advanced Topics

### Optimization Opportunities:
1. **Memory Coalescing**: Already implemented with consecutive access pattern
2. **Occupancy Optimization**: Experiment with different block sizes
3. **Shared Memory**: Not applicable for this simple operation
4. **Streams**: For overlapping computation and memory transfers

### Scaling Considerations:
- **Large Vectors**: May require multiple kernel launches
- **Memory Limitations**: GPU memory capacity constraints
- **Multi-GPU**: Distributing work across multiple devices

## Common Issues and Solutions

1. **CUDA Not Available**: Ensure proper CUDA installation and GPU drivers
2. **Memory Errors**: Check GPU memory capacity vs. vector size
3. **Performance**: Small vectors may show negative speedup due to overhead
4. **Numerical Precision**: Use consistent data types (float32 recommended)

This implementation serves as a foundation for understanding GPU programming concepts and can be extended to more complex parallel algorithms.