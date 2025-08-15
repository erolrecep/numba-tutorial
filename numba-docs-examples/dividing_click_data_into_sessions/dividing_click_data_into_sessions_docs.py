# import required libraries
import numpy as np
from numba import cuda

# Set the timeout to one hour
session_timeout = np.int64(np.timedelta64("3600", "s"))

@cuda.jit
def sessionize(user_id, timestamp, results):
    gid = cuda.grid(1)
    size = len(user_id)

    if gid >= size:
        return

    # Determine session boundaries
    is_first_datapoint = gid == 0
    if not is_first_datapoint:
        new_user = user_id[gid] != user_id[gid - 1]
        timed_out = (
            timestamp[gid] - timestamp[gid - 1] > session_timeout
        )
        is_sess_boundary = new_user or timed_out
    else:
        is_sess_boundary = True

    # Determine session labels
    if is_sess_boundary:
        # This thread marks the start of a session
        results[gid] = gid

        # Make sure all session boundaries are written
        # before populating the session id
        grid = cuda.cg.this_grid()
        grid.sync()

        look_ahead = 1
        # Check elements 'forward' of this one
        # until a new session boundary is found
        while results[gid + look_ahead] == 0:
            results[gid + look_ahead] = gid
            look_ahead += 1
            # Avoid out-of-bounds accesses by the last thread
            if gid + look_ahead == size - 1:
                results[gid + look_ahead] = gid
                break

# Generate data
ids = cuda.to_device(
    np.array(
        [
            1, 1, 1, 1, 1, 1,
            2, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4, 4, 4, 4,
        ]
    )
)
sec = cuda.to_device(
    np.array(
        [
            1, 2, 3, 5000, 5001, 5002, 1,
            2, 3, 1, 2, 5000, 5001, 10000,
            10001, 10002, 10003, 15000, 150001,
            1, 5000, 50001, 15000, 20000,
            25000, 25001, 25002, 25003,
        ],
        dtype="datetime64[ns]",
    ).astype(
        "int64"
    )  # Cast to int64 for compatibility
)
# Create a vector to hold the results
results = cuda.to_device(np.zeros(len(ids)))

# Launch the kernel
sessionize[1, len(ids)](ids, sec, results)

# Print the results
print(results.copy_to_host())
