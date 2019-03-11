from __future__ import division
import numpy as np
import scipy as sp

def construct_A(n):
    A = np.diag([2] * n)

    A += np.diag([-1] * (n - 1), k=1)
    A += np.diag([-1] * (n - 1), k=-1)

    condn_no = np.linalg.cond(A, p=2)
    return A, condn_no

if __name__ == '__main__':
    A, condn_no = construct_A(512)
    print("A = \n", A)
    print("Condition number of A  = ", condn_no)