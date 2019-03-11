from __future__ import division
import numpy as np
import scipy as sp

def back_substitute(U, bb):
    n = U.shape[1]
    x = np.zeros(n)
    for j in range(n - 1, -1, -1):   # loop backwards over columns
        if U[j, j] == 0:
            raise RuntimeError("singular matrix")
        x[j] = bb[j] / U[j, j]
        for i in range(0, j):
            bb[i] -= U[i, j] * x[j]
    return x
