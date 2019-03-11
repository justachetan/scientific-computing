import numpy as np
import matplotlib.pyplot as plt

def QR_fact(A):
	
	"""
		I spent 4 hours on this. This still does
		not work properly. Ultimately I just cries
		and left this as is in remembrance.
	"""


	if A is None:
		raise RuntimeError("A cannot be NoneType")
	
	ncols = A.shape[1]
	nrows = A.shape[0]
	
	Q = np.zeros(A.shape)
	R = np.zeros((ncols, ncols))
	for i in range(ncols):
		u_i = A[:, i]
		u_i-=np.dot(Q[:, :i], np.dot(u_i, Q[:, :i]))
		e_i = u_i / np.linalg.norm(u_i, ord=2)
		Q[:, i] = e_i
		R[:, i] = np.dot(np.dot(u_i, Q[:, :i+1]), np.diag(np.ones(ncols))[:i+1])
	return Q, R

def QR_fact_iter(A):
	if A is None:
		raise RuntimeError("A cannot be NoneType")
	
	ncols = A.shape[1]
	nrows = A.shape[0]
	
	Q = np.zeros(A.shape)
	R = np.zeros((ncols, ncols))
	
	
	for k in range(ncols):
		
		Q[:, k] = A[:, k]
		for j in range(k):
			
			R[j, k] = np.dot(Q[:, j], A[:, k])
			Q[:, k] = Q[:, k] - R[j, k] * Q[:, j]
		
		R[k, k] = np.linalg.norm(Q[:, k], ord=2)
		if R[k, k] == 0:
			raise RuntimeError("Matrix A is not full rank.")
		
		Q[:, k] = Q[:, k] / R[k, k]
		
	return Q, R

def main():
	A = np.random.random((100, 80))
	Q, R = QR_fact_iter(A)


if __name__ == '__main__':
	main()