import numpy as np

def rayleigh_iter(A, x):
	diff = 100000
	xo = x
	xk = None
	iters = 0
	while(diff > 10**(-12)):
		iters+=1

		sigma_k = np.dot(xo, np.dot(A, xo)) / np.dot(xo, xo)

		yk = np.linalg.solve(A - (sigma_k * np.diag(np.diag(np.ones(A.shape)))), xo)

		xk = yk / np.linalg.norm(yk, ord=2)

		diff = np.linalg.norm(xk - xo, ord=2) / np.linalg.norm(xo, ord=2)

		xo = xk
	return sigma_k, xk, iters

if __name__ == '__main__':
	print("Rayleigh Quotient Iteration")
	print("------------------------------\n\n")

	seeds = [1, 89, 98, 23, 88, 91, 101, 11, 17, 19]

	for i in range(10):
		A = np.array([[6, 2, 1], [2, 3, 1], [1, 1, 1]])
		np.random.seed(seeds[i])
		print('Seed:', seeds[i])
		x0 = np.random.randn(3)
		l, x, iters = rayleigh_iter(A, x0)
		
		print("Computed Eigenvalue:", l)
		print("Computed Eigenvector:\n\t", x, "'\n")
		print("Number of iterations:", iters)
		print("\n\n")