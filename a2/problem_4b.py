from __future__ import division
import numpy as np
import scipy as sp
from helper import back_substitute


def get_index_of_pivot(A, i):
    # returns the index of the row suitable for pivot in column i
    max_elem = None
    max_index = None
    for j in range(i+1, A.shape[0]):
        if max_elem == None or abs(max_elem) < abs(A[j][i]):
            max_elem = A[j][i]
            max_index = j
    return max_index

def swap_rows(A, i, j):
    # swaps rows i and j of the matrix A
    A[i], A[j] = A[j].copy(), A[i].copy()
    return A

def ge_pp(A, b):
    n = len(A)
    for k in range(0, n - 1):
        M = np.identity(A.shape[0], dtype=np.float64)
        p = get_index_of_pivot(A, k)
        if p != k:
            A = swap_rows(A, k, p)
            b = swap_rows(b, k, p)
        if A[k][k] == 0:
            continue
        for i in range(k+1, n):
            M[i][k] = -1 * (A[i][k] / A[k][k])
        U = np.array([[0] * n for i in range(n)], dtype=np.float64)
        for i in range(0, n):
            for j in range(0, n):
                U[i, j] = np.dot(M[i], A.T[j])
#         print(A, "\n", U,"\n", M)
        A = np.asarray(U)
        b_dash = np.zeros(b.shape, dtype=np.float64)
        for i in range(0, n):
            b_dash[i] = np.dot(M[i], b)

        b = b_dash
    return A, b

def main():
    A = np.array([[1, 2, 2], [4, 4, 2], [4, 6, 4]], dtype=np.float64)
    b = np.array([3, 6, 10], dtype=np.float64)
    U, b_n = ge_pp(A, b)
    x = back_substitute(U, b_n)
    print("A = \n", A)
    print("b = \n", b)
    print("Solution for Ax = b, x = \n", x)

if __name__ == '__main__':
    main()