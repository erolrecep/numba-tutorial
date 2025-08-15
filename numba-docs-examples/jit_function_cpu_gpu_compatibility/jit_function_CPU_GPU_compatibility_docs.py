# import required libraries
import numba
from numba import cuda
from math import pi

@numba.jit
def business_logic(x, y, z):
    return 4 * z * (2 * x - (4 * y) / 2 * pi)

print(business_logic(1, 2, 3))  # -126.79644737231007

X = cuda.to_device([1, 10, 234])
Y = cuda.to_device([2, 2, 4014])
Z = cuda.to_device([3, 14, 2211])
results = cuda.to_device([0.0, 0.0, 0.0])

@cuda.jit
def f(res, xarr, yarr, zarr):
    tid = cuda.grid(1)
    if tid < len(xarr):
        # The function decorated with numba.jit may be directly reused
        res[tid] = business_logic(xarr[tid], yarr[tid], zarr[tid])

f.forall(len(X))(results, X, Y, Z)
print(results)
# [-126.79644737231007, 416.28324559588634, -218912930.2987788]
