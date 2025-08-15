# Moving Average with Numba guvectorize

## Overview
This demonstrates **Numba's `guvectorize` decorator** for creating **universal functions (ufuncs)** that can operate on NumPy arrays with automatic broadcasting and vectorization.

## Key Components

**1. The `@guvectorize` Decorator:**
- **Signature**: `['void(float64[:], intp[:], float64[:])']` specifies input/output types
  - `float64[:]` = 1D array of 64-bit floats
  - `intp[:]` = array of integers (for window size)
  - `void` = function doesn't return a value (modifies output array in-place)
- **Layout**: `'(n),()->(n)'` defines the broadcasting pattern
  - `(n)` = input array of size n
  - `()` = scalar input (window size)
  - `(n)` = output array of same size as input

**2. Moving Average Algorithm:**
- **Phase 1**: For the first `window_width` elements, calculates expanding averages (1, 2, 3... elements)
- **Phase 2**: For remaining elements, uses efficient sliding window technique
  - Adds new element and subtracts the element that "falls off" the window
  - Maintains constant window size for true moving average

**3. Key Benefits of `guvectorize`:**
- **Automatic vectorization** - works on multi-dimensional arrays without explicit loops
- **Broadcasting** - automatically handles different array shapes
- **Performance** - compiled to fast machine code
- **NumPy integration** - behaves like a native NumPy function

## Important Things to Know

1. **In-place modification** - The function modifies the `out` parameter directly rather than returning values
2. **Efficient algorithm** - Uses O(1) operations per element after initial window setup
3. **Flexible input** - Can handle multi-dimensional arrays (example shows 2×10 array)
4. **Window behavior** - Early elements use expanding windows, later elements use fixed sliding windows

This is a great example of how Numba can create high-performance, vectorized functions that integrate seamlessly with NumPy's ecosystem.