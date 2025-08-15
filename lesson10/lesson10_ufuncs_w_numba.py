# import required libraries
import numpy as np
from numba import vectorize
import time

def numpy_sigmoid(x):
    return 1 / (1 + np.exp(-x))

@vectorize(['float64(float64)'], target='parallel')
def numba_sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 10_000_000)

start = time.time()
y1 = numpy_sigmoid(x)
print(f"NumPy sigmoid: {time.time() - start:.4f} sec")

start = time.time()
y2 = numba_sigmoid(x)
print(f"Numba ufunc sigmoid: {time.time() - start:.4f} sec")
