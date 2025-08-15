# import required libraries
import numpy as np
import time
from numba import njit

'''
Numba has quite a few decorators, we’ve seen @jit, but there’s also:
 - @njit - this is an alias for @jit(nopython=True) as it is so commonly used!
 - @vectorize - produces NumPy ufunc s (with all the ufunc methods supported). Docs are here.
 - @guvectorize - produces NumPy generalized ufunc s. Docs are here.
 - @stencil - declare a function as a kernel for a stencil like operation. Docs are here.
 - @jitclass - for jit aware classes. Docs are here.
 - @cfunc - declare a function for use as a native call back (to be called from C/C++ etc). Docs are here.
 - @overload - register your own implementation of a function for use in nopython mode, e.g. @overload(scipy.special.j0). Docs are here.
Extra options available in some decorators:
 - parallel = True - enable the automatic parallelization of the function.
 - fastmath = True - enable fast-math behaviour for the function.
ctypes/cffi/cython interoperability:
 - cffi - The calling of CFFI functions is supported in nopython mode.
 - ctypes - The calling of ctypes wrapped functions is supported in nopython mode.
 - Cython exported functions are callable.
'''


def ident_np_py(x):
    return np.cos(x) ** 2 + np.sin(x) ** 2

@njit
def ident_np_jit(x):
    return np.cos(x) ** 2 + np.sin(x) ** 2

def ident_loops_py(x):
    r = np.empty_like(x)
    n = len(x)
    for i in range(n):
        r[i] = np.cos(x[i]) ** 2 + np.sin(x[i]) ** 2
    return r

@njit
def ident_loops_jit(x):
    r = np.empty_like(x)
    n = len(x)
    for i in range(n):
        r[i] = np.cos(x[i]) ** 2 + np.sin(x[i]) ** 2
    return r

x = np.linspace(0, 100, 1_000_000)

# Warm up JIT
ident_np_jit(x)
ident_loops_jit(x)

start = time.time()
ident_np_py(x)
print(f"ident_np_py     : {time.time() - start:.6f} s")

start = time.time()
ident_np_jit(x)
print(f"ident_np_jit    : {time.time() - start:.6f} s")

start = time.time()
ident_loops_py(x)
print(f"ident_loops_py  : {time.time() - start:.6f} s")

start = time.time()
ident_loops_jit(x)
print(f"ident_loops_jit : {time.time() - start:.6f} s")
