import numpy as np
import matplotlib.pyplot as plt
from problem_2a import QR_fact_iter


A_1 = np.random.random((5, 5))
print("Shape of Matrix A =", A_1.shape)
Q, R = QR_fact_iter(A_1)
rel_err = np.linalg.norm((np.dot(Q, R) - A_1), ord=2) / np.linalg.norm(A_1, ord=2)
print("Relative Error =", rel_err)
print("Condition Number of A =", np.linalg.cond(A_1, p=2))
print("Condition Number of QR =", np.linalg.cond(np.dot(Q, R), p=2))
print("\n")

A_2 = np.random.random((10, 10))
print("Shape of Matrix A =", A_2.shape)
Q, R = QR_fact_iter(A_2)
rel_err = np.linalg.norm((np.dot(Q, R) - A_2), ord=2) / np.linalg.norm(A_2, ord=2)
print("Relative Error =", rel_err)
print("Condition Number of A =", np.linalg.cond(A_2, p=2))
print("Condition Number of QR =", np.linalg.cond(np.dot(Q, R), p=2))
print("\n")

A_3 = np.random.random((100, 80))
print("Shape of Matrix A =", A_3.shape)
Q, R = QR_fact_iter(A_3)
rel_err = np.linalg.norm((np.dot(Q, R) - A_3), ord=2) / np.linalg.norm(A_3, ord=2)
print("Relative Error =", rel_err)
print("Condition Number of A =", np.linalg.cond(A_2, p=2))
print("Condition Number of QR =", np.linalg.cond(np.dot(Q, R), p=2))
print("\n")