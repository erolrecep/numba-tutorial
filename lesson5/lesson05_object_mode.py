import numpy as np
import pandas as pd
from numba import jit
import time

'''
What's Happening with the Generated Code
 - go_fast function (nopython mode)
    + This function is fully JIT-compiled to machine code using Numba’s nopython mode because all operations are NumPy-based and statically typable.
    + The generated LLVM/assembly (viewable via @jit(nopython=True) and numba --annotate-html) will show efficient loops and vectorized operations.
    + Benefit: Major performance gains (often 10x–100x speedup).
    
 - use_pandas function (object mode)
    + This function is not fully JIT-compiled to machine code.
    + Even though it's decorated with @jit, it's forced into object mode via forceobj=True because Numba doesn’t understand Pandas internals.
    + The function is run through Python’s standard interpreter with some loop lifting optimizations (if any).
    + The generated LLVM/assembly (viewable via @jit(forceobj=True) and numba --annotate-html) will show Python objects and calls.
    + Benefit: Minor performance gains (if any) due to loop lifting.
    + The generated code will essentially wrap Python objects; there’s no significant speedup.
    
--> The Numba @jit decorator fundamentally operates in two compilation modes, nopython mode and object mode. 
In the go_fast example above, the @jit decorator defaults to operating in nopython mode. 
The behaviour of the nopython compilation mode is to essentially compile the decorated function so that it will run entirely without the involvement of the Python interpreter. 
This is the recommended and best-practice way to use the Numba jit decorator as it leads to the best performance.
'''

x = np.arange(100).reshape(10, 10)

@jit
def go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

start = time.perf_counter()
out1 = go_fast(x)
elapsed1 = time.perf_counter() - start
print(out1)
print(f"Numba JIT (go_fast) time: {elapsed1:.6f} s")

x_dict = {'a': [1, 2, 3], 'b': [20, 30, 40]}

@jit(forceobj=True, looplift=True)
def use_pandas(a):
    df = pd.DataFrame.from_dict(a)
    df += 1
    return df.cov()

start = time.perf_counter()
out2 = use_pandas(x_dict)
elapsed2 = time.perf_counter() - start
print(out2)
print(f"Numba object mode (use_pandas) time: {elapsed2:.6f} s")
