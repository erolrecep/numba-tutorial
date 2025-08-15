# import required libraries
import numpy as np
from numba import cuda

# Use an odd problem size.
# This is so there can be an element truly in the "middle" for symmetry.
size = 1001
data = np.zeros(size)

# Middle element is made very hot
data[500] = 10000
buf_0 = cuda.to_device(data)

# This extra array is used for synchronization purposes
buf_1 = cuda.device_array_like(buf_0)

niter = 10000

@cuda.jit
def solve_heat_equation(buf_0, buf_1, timesteps, k):
    i = cuda.grid(1)

    # Don't continue if our index is outside the domain
    if i >= len(buf_0):
        return

    # Prepare to do a grid-wide synchronization later
    grid = cuda.cg.this_grid()

    for step in range(timesteps):
        # Select the buffer from the previous timestep
        if (step % 2) == 0:
            data = buf_0
            next_data = buf_1
        else:
            data = buf_1
            next_data = buf_0

        # Get the current temperature associated with this point
        curr_temp = data[i]

        # Apply formula from finite difference equation
        if i == 0:
            # Left wall is held at T = 0
            next_temp = curr_temp + k * (data[i + 1] - (2 * curr_temp))
        elif i == len(data) - 1:
            # Right wall is held at T = 0
            next_temp = curr_temp + k * (data[i - 1] - (2 * curr_temp))
        else:
            # Interior points are a weighted average of their neighbors
            next_temp = curr_temp + k * (
                data[i - 1] - (2 * curr_temp) + data[i + 1]
            )

        # Write new value to the next buffer
        next_data[i] = next_temp

        # Wait for every thread to write before moving on
        grid.sync()

# call the kernel
solve_heat_equation.forall(len(data))(
    buf_0, buf_1, niter, 0.25
)
