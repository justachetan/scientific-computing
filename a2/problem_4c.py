from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from problem_4a import ge_nop
from problem_4b import ge_pp
from helper import back_substitute

# def bmatrix(a):
#     """Returns a LaTeX bmatrix

#     :a: numpy array
#     :returns: LaTeX bmatrix as a string
#     """
#     if len(a.shape) > 2:
#         raise ValueError('bmatrix can at most display two dimensions')
#     lines = str(a).replace('[', '').replace(']', '').splitlines()
#     rv = [r'\begin{bmatrix}']
#     rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
#     rv +=  [r'\end{bmatrix}']
#     return '\n'.join(rv)




n = 100
A_1 = np.random.randn(n, n)
A_2 = np.array([[0] * n for i in range(n)], dtype=np.float64)
A_3 = np.array([[0] * n for i in range(n)], dtype=np.float64)
for i in range(n):
    for j in range(n):
        if (i + 2)%n == j:
            A_2[i, j] = 5
        else:
            A_2[i, j] = 10**(-3)
        A_3[i, j] = 1 / ((1 + abs((i + 1)%n - j))**4)

np.random.seed(2018)
x_1, x_2, x_3 = np.random.rand(n, 3).T

b_1 = np.zeros((n,), dtype=np.float64)
b_2 = np.zeros((n,), dtype=np.float64)
b_3 = np.zeros((n,), dtype=np.float64)
for i in range(n):
    b_1[i] = np.dot(A_1[i], x_1)
    b_2[i] = np.dot(A_2[i], x_2)
    b_3[i] = np.dot(A_3[i], x_3)

cno_1 = np.linalg.cond(A_1, p=2)
cno_2 = np.linalg.cond(A_2, p=2)
cno_3 = np.linalg.cond(A_3, p=2)


x_err_nop_1 = None
x_err_nop_2 = None
x_err_nop_3 = None
x_rerr_nop_1 = None
x_rerr_nop_2 = None
x_rerr_nop_3 = None
res_nop_1 = None
res_nop_2 = None
res_nop_3 = None
rres_nop_1 = None
rres_nop_2 = None
rres_nop_3 = None


try:
    x_sol_nop_1 = back_substitute(ge_nop(A_1, b_1)[0], ge_nop(A_1, b_1)[1])
    x_err_nop_1 = np.linalg.norm(x_sol_nop_1 - x_1, ord=2)
    x_rerr_nop_1 = x_err_nop_1 / np.linalg.norm(x_1, ord=2)
    res_nop_1 = np.linalg.norm(b_1 - np.array([np.dot(A_1[i], x_sol_nop_1) for i in range(n)],\
                                              dtype=np.float64), ord=2)
    rres_nop_1 = res_nop_1 / (np.linalg.norm(A_1, ord=2) * np.linalg.norm(x_1, ord=2))
    
except RuntimeError as e:
    x_sol_nop_1 = np.nan
    x_err_nop_1 = np.nan
    res_nop_1 = np.nan
    rres_nop_1 = np.nan
    x_rerr_nop_1 = np.nan
    print(e)
    print("Faulty Matrix is, A_1 = \n", A_1)

try:    
    x_sol_nop_2 = back_substitute(ge_nop(A_2, b_2)[0], ge_nop(A_2, b_2)[1])
    x_err_nop_2 = np.linalg.norm(x_sol_nop_2 - x_2, ord=2)
    x_rerr_nop_2 = x_err_nop_2 / np.linalg.norm(x_2, ord=2)
    res_nop_2 = np.linalg.norm(b_2 - np.array([np.dot(A_2[i], x_sol_nop_2) for i in range(n)],\
                                              dtype=np.float64), ord=2)
    rres_nop_2 = res_nop_2 / (np.linalg.norm(A_2, ord=2) * np.linalg.norm(x_2, ord=2))
    
except RuntimeError as e:
    x_sol_nop_2 = np.nan
    x_err_nop_2 = np.nan
    x_rerr_nop_2 = np.nan
    res_nop_2 = np.nan
    rres_nop_2 = np.nan
    print(e)
    print("Faulty Matrix is, A_2 = \n", A_2)
    # print(bmatrix(A_2))

try:
    x_sol_nop_3 = back_substitute(ge_nop(A_3, b_3)[0], ge_nop(A_3, b_3)[1])
    x_err_nop_3 = np.linalg.norm(x_sol_nop_3 - x_3, ord=2)
    x_rerr_nop_3 = x_err_nop_3 / np.linalg.norm(x_3, ord=2)
    res_nop_3 = np.linalg.norm(b_3 - np.array([np.dot(A_3[i], x_sol_nop_3) for i in range(n)],\
                                              dtype=np.float64), ord=2)
    rres_nop_3 = res_nop_3 / (np.linalg.norm(A_3, ord=2) * np.linalg.norm(x_3, ord=2))
    
except RuntimeError as e:
    x_sol_nop_3 = np.nan
    x_err_nop_3 = np.nan
    x_rerr_nop_3 = np.nan
    res_nop_3 = np.nan
    rres_nop_3 = np.nan
    print(e)
    print("Faulty Matrix is, A_3 = \n", A_3)



U_1, b_n1 = ge_pp(A_1, b_1)
U_2, b_n2 = ge_pp(A_2, b_2)
U_3, b_n3 = ge_pp(A_3, b_3)

x_sol_pp_1 = back_substitute(U_1, b_n1)
x_sol_pp_2 = back_substitute(U_2, b_n2)
x_sol_pp_3 = back_substitute(U_3, b_n3)

x_err_pp_1 = np.linalg.norm(x_sol_pp_1 - x_1, ord=2)
x_err_pp_2 = np.linalg.norm(x_sol_pp_2 - x_2, ord=2)
x_err_pp_3 = np.linalg.norm(x_sol_pp_3 - x_3, ord=2)

x_rerr_pp_1 = x_err_pp_1 / np.linalg.norm(x_1, ord=2)
x_rerr_pp_2 = x_err_pp_2 / np.linalg.norm(x_2, ord=2)
x_rerr_pp_3 = x_err_pp_3 / np.linalg.norm(x_3, ord=2)


res_pp_1 = np.linalg.norm(b_1 - np.array([np.dot(A_1[i], x_sol_pp_1) for i in range(n)],\
                                              dtype=np.float64), ord=2)
res_pp_2 = np.linalg.norm(b_2 - np.array([np.dot(A_2[i], x_sol_pp_2) for i in range(n)],\
                                              dtype=np.float64), ord=2)
res_pp_3 = np.linalg.norm(b_3 - np.array([np.dot(A_3[i], x_sol_pp_3) for i in range(n)],\
                                              dtype=np.float64), ord=2)

rres_pp_1 = res_pp_1 / (np.linalg.norm(A_1, ord=2) * np.linalg.norm(x_1, ord=2))
rres_pp_2 = res_pp_2 / (np.linalg.norm(A_2, ord=2) * np.linalg.norm(x_2, ord=2))
rres_pp_3 = res_pp_3 / (np.linalg.norm(A_3, ord=2) * np.linalg.norm(x_3, ord=2))


x_1_np = np.linalg.solve(A_1, b_1)
x_2_np = np.linalg.solve(A_2, b_2)
x_3_np = np.linalg.solve(A_3, b_3)

x_err_np_1 = np.linalg.norm(x_1_np - x_1, ord=2)
x_err_np_2 = np.linalg.norm(x_2_np - x_2, ord=2)
x_err_np_3 = np.linalg.norm(x_3_np - x_3, ord=2)

x_rerr_np_1 = x_err_np_1 / np.linalg.norm(x_1, ord=2)
x_rerr_np_2 = x_err_np_2 / np.linalg.norm(x_2, ord=2)
x_rerr_np_3 = x_err_np_3 / np.linalg.norm(x_3, ord=2)

res_np_1 = np.linalg.norm(b_1 - np.array([np.dot(A_1[i], x_1_np) for i in range(n)],\
                                              dtype=np.float64), ord=2)
res_np_2 = np.linalg.norm(b_2 - np.array([np.dot(A_2[i], x_2_np) for i in range(n)],\
                                              dtype=np.float64), ord=2)
res_np_3 = np.linalg.norm(b_3 - np.array([np.dot(A_3[i], x_3_np) for i in range(n)],\
                                              dtype=np.float64), ord=2)

rres_np_1 = res_np_1 / (np.linalg.norm(A_1, ord=2) * np.linalg.norm(x_1, ord=2))
rres_np_2 = res_np_1 / (np.linalg.norm(A_2, ord=2) * np.linalg.norm(x_2, ord=2))
rres_np_3 = res_np_1 / (np.linalg.norm(A_3, ord=2) * np.linalg.norm(x_3, ord=2))


index = [("Error", "ge_nop"), ("Error", "ge_pp"), ("Error", "np.solve"),\
         ("Residual", "ge_nop"), ("Residual", "ge_pp"), ("Residual", "np.solve"),\
         ("Relative Error", "ge_nop"), ("Relative Error", "ge_pp"), ("Relative Error", "np.solve"),\
         ("Relative Residual", "ge_nop"), ("Relative Residual", "ge_pp"), ("Relative Residual", "np.solve")]

index = pd.MultiIndex.from_tuples(index, names=['Statistic', 'Method'])

matrix = {'A_1, x_1': [x_err_nop_1, x_err_pp_1, x_err_np_1,\
                       res_nop_1, res_pp_1, res_np_1,\
                       x_rerr_nop_1, x_rerr_pp_1, x_rerr_np_1,\
                       rres_nop_1, rres_pp_1, rres_pp_1],\
          'A_2, x_2': [x_err_nop_2, x_err_pp_2, x_err_np_2,\
                       res_nop_2, res_pp_2, res_np_2,\
                       x_rerr_nop_2, x_rerr_pp_2, x_rerr_np_2,\
                       rres_nop_2, rres_pp_2, rres_pp_2], \
          'A_3, x_3': [x_err_nop_3, x_err_pp_3, x_err_np_3,\
                       res_nop_3, res_pp_3, res_np_3,\
                       x_rerr_nop_3, x_rerr_pp_3, x_rerr_np_3,\
                       rres_nop_1, rres_pp_1, rres_pp_1]}


df = pd.DataFrame(matrix, index = index)
df.columns.name = "Input"
df = df.T
df['Condition number'] = pd.Series([cno_1, cno_2, cno_3], index=df.T.columns)
df = df.T

print('\n\n')
print('\t\tAnalysis Table\n')

print(df.to_string())
# print(df.to_latex())
