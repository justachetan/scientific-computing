from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from helper import back_substitute

def ge_nop(A, b):
    n = len(A)
    for k in range(0, n-1):
        
        M = np.identity(A.shape[0])
        if A[k][k] == 0:
            raise RuntimeError("pivot element 0")
        for i in range(k+1, n):
            M[i][k] = -1 * (A[i][k] / A[k][k])
        U = np.array([[0] * n for i in range(n)], dtype=np.float64)
#         print(A,"\n", M)
        for i in range(0, n):
            for j in range(0, n):
                U[i, j] = np.dot(M[i], A.T[j])
        A = np.asarray(U)
#         print(A,"\n", M)
        b_dash = np.zeros(b.shape)
        for i in range(0, n):
            b_dash[i] = np.dot(M[i], b)
        b = b_dash
    return A, b


def main():
    try:
        A = np.array([[1, 2, 2], [4, 4, 2], [4, 6, 4]], dtype=np.float64)
        b = np.array([3, 6, 10], dtype=np.float64)
        U, b_n = ge_nop(A, b)
        x = back_substitute(U, b_n)
        print("A = \n", A)
        print("b = \n", b)
        print("Solution for Ax = b, x = \n", x.T)
    except RuntimeError as e:
        print(e)
        print("Faulty Matrix is, \n", A)

if __name__ == '__main__':
    main()

