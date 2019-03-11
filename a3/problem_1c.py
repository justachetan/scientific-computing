import numpy as np
from problem_1b import pow_iter


def deflate(A, x_1):
	k = 0
	for i in range(len(x_1)):
		if x_1[i] == 1: k = i
	u_1 = A.T[:, k]
	A_new = A - np.outer(x_1, u_1)

	x2, lamda = pow_iter(A_new, np.random.random(3))
	return lamda

def main():
	A = np.array([[2, 3, 2], [10, 3, 4], [3, 6, 1]])
	x_0 = np.array([0, 0, 1])
	x, lamda = pow_iter(A.copy(), x_0)
	lamda_2 = deflate(A.copy(), x)
	print("Absolute Value of Largest Eigenvalue:", lamda)
	print("Normalized Eigenvector:", x)
	print("Absolute Value of Second - largest Eigenvalue:", lamda_2)

if __name__ == '__main__':
	main()