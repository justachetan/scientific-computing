import numpy as np
from problem_3a import inverse_iteration
from problem_3b import rayleigh_iter

def main():
	print("Inverse Iteration")
	print("-------------------\n\n")
	
	A = np.array([[6, 2, 1], [2, 3, 1], [1, 1, 1]])
	x0 = np.array([1, 4, 2])
	print("Initial Array:\n\t", x0, "'\n")
	l, x, iters = inverse_iteration(A, x0)
	
	print("Computed Eigenvalue:", l)
	print("Computed Eigenvector:\n\t", x, "'\n")
	print("Number of iterations:", iters)

	print("\n\n")

	print("Rayleigh Quotient Iteration")
	print("------------------------------\n\n")


	A = np.array([[6, 2, 1], [2, 3, 1], [1, 1, 1]])
	x0 = np.array([1, 4, 2])
	print("Initial Array:\n\t", x0, "'\n")
	l_r, x_r, iters_r = rayleigh_iter(A, x0)
	
	print("Computed Eigenvalue:", l_r)
	print("Computed Eigenvector:\n\t", x_r, "'\n")
	print("Number of iterations:", iters_r)
	print("\n\n")

	print("Picking Rayleigh Quotient Iteration result as true value...\n\n")

	rel_err_vec = np.linalg.norm(x - x_r, ord=2) / np.linalg.norm(x_r, ord=2)
	rel_err_val = np.abs(l - l_r) / np.abs(l_r)

	print("Relative Error for Eigenvector:", rel_err_vec)
	print("Relative Error for Eigenvalue:", rel_err_val)


if __name__ == '__main__':
	main()