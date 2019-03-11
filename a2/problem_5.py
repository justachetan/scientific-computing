from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd


def matmul(A, B):
    # multiply two 2-D square matrices together
    n = A.shape[0]
    U = np.array([[0] * n for i in range(n)], dtype=np.float64)
    for i in range(0, n):
        for j in range(0, n):
            U[i, j] = np.dot(A[i], B.T[j])
    return U

def rel_residual(A, x_cap, x, b):
    # returns 2-norm relative residual of a linear system
    res = np.linalg.norm(b - np.array([np.dot(A[i], x_cap) for i in range(n)]), ord=2)
    rel_res = res / (np.linalg.norm(A, ord=2) * np.linalg.norm(x, ord=2))
    return rel_res

def rel_error(x_cap, x):
    # returns 2-norm relative error
    return np.linalg.norm(x_cap - x, ord=2) / np.linalg.norm(x, ord=2)


n = 100
np.random.seed(1729)
A = np.random.randn(n, n)

x = np.ones(n)
b = np.array([np.dot(A[i], x) for i in range(n)])

DA = A
Db = b
x_cap = np.linalg.solve(DA, Db)
rel_err_1 = rel_error(x_cap, x)
rel_res_1 = rel_residual(DA, x_cap, x, Db)
cno_1 = np.linalg.cond(DA, p=2)


D = np.diag(2 * np.ones(n))
DA = matmul(D, A)
Db = np.array([np.dot(D[i], b) for i in range(n)])
x_cap = np.linalg.solve(DA, Db)
rel_err_2 = rel_error(x_cap, x)
rel_res_2 = rel_residual(DA, x_cap, x, Db)
cno_2 = np.linalg.cond(DA, p=2)


D = np.diag(np.linspace(1, 100, 100))
DA = matmul(D, A)
Db = np.array([np.dot(D[i], b) for i in range(n)])
x_cap = np.linalg.solve(DA, Db)
rel_err_3 = rel_error(x_cap, x)
rel_res_3 = rel_residual(DA, x_cap, x, Db)
cno_3 = np.linalg.cond(DA, p=2)


D = np.diag(np.linspace(1, 10000, 100))
DA = matmul(D, A)
Db = np.array([np.dot(D[i], b) for i in range(n)])
x_cap = np.linalg.solve(DA, Db)
rel_err_4 = rel_error(x_cap, x)
rel_res_4 = rel_residual(DA, x_cap, x, Db)
cno_4 = np.linalg.cond(DA, p=2)


D = np.diag(2**-np.arange(-n//2, n//2, dtype=np.float64))
DA = matmul(D, A)
Db = np.array([np.dot(D[i], b) for i in range(n)])
x_cap = np.linalg.solve(DA, Db)
rel_err_5 = rel_error(x_cap, x)
rel_res_5 = rel_residual(DA, x_cap, x, Db)
cno_5 = np.linalg.cond(DA, p=2)


data = pd.DataFrame({"(i)" : [rel_err_1, rel_res_1, cno_1], "(ii)" : [rel_err_2, rel_res_2, cno_2],\
              "(iii)" : [rel_err_3, rel_res_3, cno_3], "(iv)" : [rel_err_4, rel_res_4, cno_4],\
              "(v)" : [rel_err_5, rel_res_5, cno_5]}, index=["Relative Error", "Relative Residual",\
                                                             "Condition Number"]).T
data.index.name = "Part"
data.columns.name = "Statistic"


print(data.to_string())

# print(data.to_latex())