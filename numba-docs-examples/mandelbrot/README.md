# Mandelbrot Set with Numba

## Overview
This is a **Mandelbrot set fractal generator** that uses **Numba JIT compilation** to accelerate the computation on CPU.

## Key Components

**1. Mandelbrot Set Calculation (`mandel` function):**
- Takes a complex number (x, y coordinates) and tests if it belongs to the Mandelbrot set
- Uses the iterative formula: `z = z² + c` where c is the complex number
- If the magnitude of z exceeds 2 (squared magnitude ≥ 4) within `max_iters` iterations, the point escapes
- Returns the iteration count when it escapes, or 255 if it doesn't escape (indicating it's likely in the set)

**2. Fractal Image Generation (`create_fractal` function):**
- Maps a rectangular region of the complex plane (-2 to 1 on real axis, -1 to 1 on imaginary axis) to pixel coordinates
- For each pixel, calculates the corresponding complex number and calls `mandel`
- Stores the iteration count as the pixel color value

**3. Performance Optimization:**
- Both functions use `@jit(nopython=True)` decorator from Numba
- This compiles the Python code to optimized machine code for much faster execution

**4. Execution:**
- Creates a 1000×1500 pixel image array
- Times the fractal generation (20 iterations max)
- Prints the execution time
- Optionally displays the result using matplotlib if available

The result is a classic Mandelbrot set visualization where different colors represent how quickly points escape to infinity, creating the characteristic fractal patterns and boundaries.
