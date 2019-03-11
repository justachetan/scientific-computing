import numpy as np

def inverse_iteration(A, x, shift=2):
	diff = 10000
	diffn = 0
	xn = None
	xo = x
	lamda = 0
	iters = 0
	A = A - shift * np.diag(np.diag(np.ones(A.shape)))
	while(diff > 10**(-12)):
		iters+=1
		yn = np.linalg.solve(A, xo)
		xn = yn / np.linalg.norm(yn, ord=2)
		lamda = np.linalg.norm(yn, ord=2)
		diff = np.linalg.norm((xn - xo), ord=2) / np.linalg.norm(xo, ord=2)
		xo = xn

	return 1/(lamda) + shift, xn, iters

if __name__ == '__main__':
	print("Inverse Iteration")
	print("-------------------\n\n")

	seeds = [1, 89, 98, 23, 88, 91, 101, 11, 17, 19]

	for i in range(10):
		A = np.array([[6, 2, 1], [2, 3, 1], [1, 1, 1]])
		np.random.seed(seeds[i])
		print('Seed:', seeds[i])
		x0 = np.random.random((3,))
		l, x, iters = inverse_iteration(A, x0)
		print("Computed Eigenvalue:", l)
		print("Computed Eigenvector:\n\t", x, "'\n")
		print("Number of iterations:", iters)
		print("\n\n")


