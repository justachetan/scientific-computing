import numpy as np
from problem_3a import inverse_iteration
from problem_3b import rayleigh_iter

def main():
	print("Inverse Iteration")
	print("-------------------\n\n")

	seeds = [1, 89, 98, 23, 88, 91, 101, 11, 17, 19]

	for i in range(10):
		A = np.array([[6, 2, 1], [2, 3, 1], [1, 1, 1]])
		np.random.seed(seeds[i])
		print('Seed:', seeds[i])
		x0 = np.random.random((3,))
		l, x, iters = inverse_iteration(A, x0)
		vals, vecs = np.linalg.eig(A)
		rel_err_vals = np.min(np.abs(vals - l) / vals[np.argmin(np.abs(vals - l))])
		print("Computed Eigenvalue:", l)
		print("Computed Eigenvector:\n\t", x, "'\n")
		rel_err_vecs = np.linalg.norm(x - vecs[:, np.argmin(np.abs(vals - l))], ord=2) / np.linalg.norm(vecs[:, np.argmin(np.abs(vals - l))]) 
		print("Relative Error in Eigenvalues:", rel_err_vals)
		print("Relative Error in Eigenvectors:", rel_err_vecs)
		print("Number of iterations:", iters)
		print("\n\n")


	print("Rayleigh Quotient Iteration")
	print("------------------------------\n\n")

	seeds = [1, 89, 98, 23, 88, 91, 101, 11, 17, 19]

	for i in range(10):
		A = np.array([[6, 2, 1], [2, 3, 1], [1, 1, 1]])
		np.random.seed(seeds[i])
		print('Seed:', seeds[i])
		x0 = np.random.randn(3)
		l, x, iters = rayleigh_iter(A, x0)
		vals, vecs = np.linalg.eig(A)
		rel_err_vals = np.min(np.abs(vals - l) / vals[np.argmin(np.abs(vals - l))])
		
		rel_err_vecs = np.linalg.norm(x - vecs[:, np.argmin(np.abs(vals - l))], ord=2) / np.linalg.norm(vecs[:, np.argmin(np.abs(vals - l))]) 
		print("Computed Eigenvalue:", l)
		print("Computed Eigenvector:\n\t", x, "'\n")
		print("Relative Error in Eigenvalues:", rel_err_vals)
		print("Relative Error in Eigenvectors:", rel_err_vecs)
		print("Number of iterations:", iters)
		print("\n\n")


if __name__ == '__main__':
	main()
	