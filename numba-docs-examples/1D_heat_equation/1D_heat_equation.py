# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import cuda

@cuda.jit
def solve_heat_equation(buf_0, buf_1, timesteps, k):
    """
    CUDA kernel for solving 1D heat equation using finite difference method.

    Heat equation: ∂u/∂t = α * ∂²u/∂x²
    Finite difference: u[i]^(n+1) = u[i]^n + k * (u[i-1]^n - 2*u[i]^n + u[i+1]^n)

    Args:
        buf_0, buf_1: Double buffers for current and next timestep
        timesteps: Number of time iterations
        k: Thermal diffusivity coefficient (must satisfy stability: k ≤ 0.5)
    """
    i = cuda.grid(1)

    # Don't continue if our index is outside the domain
    if i >= len(buf_0):
        return

    # Prepare to do a grid-wide synchronization later
    grid = cuda.cg.this_grid()

    for step in range(timesteps):
        # Double buffering: alternate between buf_0 and buf_1
        if (step % 2) == 0:
            data = buf_0        # Current timestep data
            next_data = buf_1   # Next timestep data
        else:
            data = buf_1        # Current timestep data
            next_data = buf_0   # Next timestep data

        # Get the current temperature at this spatial point
        curr_temp = data[i]

        # Apply finite difference formula with boundary conditions
        if i == 0:
            # Left boundary: Dirichlet condition (T = 0)
            # Only consider right neighbor since left boundary is fixed
            next_temp = curr_temp + k * (data[i + 1] - (2 * curr_temp))
        elif i == len(data) - 1:
            # Right boundary: Dirichlet condition (T = 0)
            # Only consider left neighbor since right boundary is fixed
            next_temp = curr_temp + k * (data[i - 1] - (2 * curr_temp))
        else:
            # Interior points: standard finite difference stencil
            # Second derivative approximation: (u[i-1] - 2*u[i] + u[i+1]) / Δx²
            next_temp = curr_temp + k * (
                data[i - 1] - (2 * curr_temp) + data[i + 1]
            )

        # Write new temperature value to the next timestep buffer
        next_data[i] = next_temp

        # Grid-wide synchronization: wait for all threads before next iteration
        # This ensures all threads complete current timestep before proceeding
        grid.sync()


def solve_heat_equation_cpu(initial_data, timesteps, k):
    """
    CPU reference implementation of 1D heat equation solver.
    Used for validation and performance comparison.
    """
    data = initial_data.copy()
    size = len(data)

    for step in range(timesteps):
        new_data = np.zeros_like(data)

        for i in range(size):
            if i == 0:
                # Left boundary condition
                new_data[i] = data[i] + k * (data[i + 1] - (2 * data[i]))
            elif i == size - 1:
                # Right boundary condition
                new_data[i] = data[i] + k * (data[i - 1] - (2 * data[i]))
            else:
                # Interior points
                new_data[i] = data[i] + k * (data[i - 1] - (2 * data[i]) + data[i + 1])

        data = new_data

    return data


def create_initial_condition(size, heat_source_pos=None, initial_temp=10000):
    """
    Create initial temperature distribution.

    Args:
        size: Number of spatial points
        heat_source_pos: Position of heat source (default: middle)
        initial_temp: Initial temperature at heat source
    """
    data = np.zeros(size, dtype=np.float32)

    if heat_source_pos is None:
        heat_source_pos = size // 2

    # Set initial hot spot
    data[heat_source_pos] = initial_temp

    return data


def check_stability(k):
    """
    Check numerical stability condition for explicit finite difference scheme.
    For stability: k = α * dt / dx² ≤ 0.5
    """
    stability_limit = 0.5
    is_stable = k <= stability_limit

    print(f"Stability Analysis:")
    print(f"  Diffusion coefficient k = {k}")
    print(f"  Stability limit = {stability_limit}")
    print(f"  Scheme is {'STABLE' if is_stable else 'UNSTABLE'}")

    if not is_stable:
        print(f"  WARNING: k > {stability_limit} may cause numerical instability!")

    return is_stable


def main():
    # Problem parameters
    size = 1001  # Odd size for symmetric middle element
    niter = 10000  # Number of time steps
    k = 0.25  # Thermal diffusivity coefficient

    print("1D Heat Equation Solver")
    print("=" * 50)
    print(f"Grid size: {size}")
    print(f"Time steps: {niter}")
    print(f"Diffusion coefficient: {k}")

    # Check numerical stability
    is_stable = check_stability(k)
    if not is_stable:
        print("Proceeding anyway for demonstration...")

    print()

    try:
        # Create initial condition
        initial_data = create_initial_condition(size)
        print(f"Initial condition: Hot spot at position {size//2} with T = {initial_data.max()}")

        # CPU computation for validation (smaller problem for speed)
        print("\n--- CPU Reference Computation ---")
        cpu_size = min(101, size)  # Use smaller size for CPU
        cpu_initial = create_initial_condition(cpu_size)
        cpu_niter = min(1000, niter)  # Fewer iterations for CPU

        start_time = time.time()
        solve_heat_equation_cpu(cpu_initial, cpu_niter, k)
        cpu_time = time.time() - start_time
        print(f"CPU time ({cpu_size} points, {cpu_niter} steps): {cpu_time:.4f} seconds")

        # GPU computation
        print("\n--- GPU Computation ---")

        # Transfer data to GPU
        buf_0 = cuda.to_device(initial_data)
        buf_1 = cuda.device_array_like(buf_0)

        # Execute GPU kernel
        start_time = time.time()
        solve_heat_equation.forall(len(initial_data))(buf_0, buf_1, niter, k)
        cuda.synchronize()  # Wait for GPU to complete
        gpu_time = time.time() - start_time

        # Retrieve results from GPU
        # Final result is in buf_0 if niter is even, buf_1 if odd
        if niter % 2 == 0:
            final_result = buf_0.copy_to_host()
        else:
            final_result = buf_1.copy_to_host()

        print(f"GPU time ({size} points, {niter} steps): {gpu_time:.4f} seconds")

        # Performance comparison
        if cpu_time > 0:
            # Scale comparison based on problem size difference
            scale_factor = (size / cpu_size) * (niter / cpu_niter)
            estimated_cpu_time = cpu_time * scale_factor
            speedup = estimated_cpu_time / gpu_time
            print(f"Estimated speedup: {speedup:.2f}x")

        # Display results
        print("\n--- Results ---")
        print(f"Final maximum temperature: {final_result.max():.2f}")
        print(f"Final minimum temperature: {final_result.min():.2f}")
        print(f"Temperature at center: {final_result[size//2]:.2f}")

        # Visualization
        print("\n--- Generating Visualization ---")
        plt.figure(figsize=(12, 8))

        # Plot initial and final temperature distributions
        x = np.linspace(0, 1, size)

        plt.subplot(2, 1, 1)
        plt.plot(x, initial_data, 'r-', linewidth=2, label='Initial')
        plt.plot(x, final_result, 'b-', linewidth=2, label=f'Final (t={niter})')
        plt.xlabel('Position')
        plt.ylabel('Temperature')
        plt.title('1D Heat Equation: Temperature Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot temperature evolution at center point
        plt.subplot(2, 1, 2)
        # For demonstration, show exponential decay (theoretical)
        t = np.linspace(0, niter, 100)
        theoretical = initial_data.max() * np.exp(-k * t / 100)  # Approximate decay
        plt.plot(t, theoretical, 'g--', linewidth=2, label='Theoretical decay')
        plt.scatter([0, niter], [initial_data.max(), final_result[size//2]],
                   c=['red', 'blue'], s=100, zorder=5, label='Simulation points')
        plt.xlabel('Time Step')
        plt.ylabel('Center Temperature')
        plt.title('Temperature Evolution at Center')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('heat_equation_results.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("Visualization saved as 'heat_equation_results.png'")

    except Exception as e:
        print(f"Error during execution: {e}")
        print("Make sure you have a CUDA-capable GPU and proper CUDA installation.")
        print("Also ensure matplotlib is installed for visualization.")


if __name__ == "__main__":
    main()
