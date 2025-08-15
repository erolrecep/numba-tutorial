from numba import njit, typeof

@njit
def sum_of_squares_numba(arr):
    total = 0.0
    for x in arr:
        total += x * x
    return total

import numpy as np
arr = np.random.rand(10_000_000).astype(np.float64)
sum_of_squares_numba(arr)  # triggers compilation

# Use typeof to get exact type
sig = (typeof(arr),)

llvm_code = sum_of_squares_numba.inspect_llvm(sig)
asm_code = sum_of_squares_numba.inspect_asm(sig)

with open("sum_of_squares.ll", "w") as f:
    f.write(llvm_code)

with open("sum_of_squares.asm", "w") as f:
    f.write(asm_code)

print("LLVM and ASM code saved.")

