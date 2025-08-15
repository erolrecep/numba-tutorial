import numpy as np
from numba import guvectorize


@guvectorize(['void(float64[:], intp[:], float64[:])'], '(n),()->(n)')
def move_mean(a, window_arr, out):
    """
    Compute moving average using a sliding window.

    Parameters:
    a: input array
    window_arr: array containing window width
    out: output array for results
    """
    window_width = window_arr[0]
    asum = 0.0
    count = 0

    # Calculate initial window averages
    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count

    # Calculate sliding window averages
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / count


# Example usage
arr = np.arange(20, dtype=np.float64).reshape(2, 10)
print("Input array:")
print(arr)
print("\nMoving average (window=3):")
print(move_mean(arr, 3))
