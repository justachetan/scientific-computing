import numpy as np
import numpy.linalg as npla

def qr_iteration(A, tol):
    # Your implementation goes here
    
    for i in range(A.shape[1] - 1, 0, -1):
        while npla.norm(A[i - 1, :i - 1], ord=2) > tol:
            sigma = A[i, i]
            Q, R = npla.qr(A - sigma * np.diag(np.diag(np.ones(A.shape))))
            A = np.dot(R, Q) + sigma * np.diag(np.diag(np.ones(A.shape)))
    
    return np.diag(A)

def main():
	
	tol = 10 ** (-16)
	A_1 = np.array([[2, 3, 2], [10, 3, 4], [3, 6, 1]])
	eigenvalues_1 = qr_iteration(A_1.copy(), tol)
	print("Matrix:\n", A_1)
	print("\n")
	print ("Computed eigenvalues: ", eigenvalues_1)
	print ("Actual eigenvalues: ", np.linalg.eigvals(A_1))

	print("\n\n")

	tol = 10 ** (-16)
	A_2 = np.array([[6, 2, 1], [2, 3, 1], [1, 1, 1]])
	eigenvalues_2 = qr_iteration(A_2.copy(), tol)
	print("Matrix:\n", A_2)
	print("\n")
	print ("Computed eigenvalues: ", eigenvalues_2)
	print ("Actual eigenvalues: ", np.linalg.eigvals(A_2))

if __name__ == '__main__':
	main()
        
	