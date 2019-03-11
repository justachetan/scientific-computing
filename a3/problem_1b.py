import numpy as np

def pow_iter(A, x_0, tol=10**(-16), max_iter=100):
    y = None
    xo = x_0.copy()
    lamdao = 1
    diff = 10000
    lamdan = 0
    while diff > tol:
        y = np.dot(A, xo)
        xn = y / np.linalg.norm(y, ord=np.inf)
        lamdan = np.linalg.norm(y, ord=np.inf)
        diff = np.abs(lamdao - lamdan) / np.abs(lamdao)
        lamdao = lamdan
        xo = xn
    return xn, lamdao

def main():
	A = np.array([[2, 3, 2], [10, 3, 4], [3, 6, 1]])
	x_0 = np.array([0, 0, 1])
	x, lamda = pow_iter(A.copy(), x_0)
	print("Absolute Value of Largest Eigenvalue:", lamda)
	print("Normalized Eigenvector:", x)

if __name__ == '__main__':
	main()