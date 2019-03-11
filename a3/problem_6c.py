import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
from problem_2a import QR_fact_iter
from problem_6b import lancoiz



def main():

	B = np.random.random((100, 100))
	Q, R = QR_fact_iter(B)
	D = np.diag(np.arange(1, Q.shape[1] + 1))
	A = np.dot(np.dot(Q, D), Q.T)
	Q, H, rvals = lancoiz(A)

	plt.xlabel("Ritz Values")
	plt.ylabel("Number of iterations")
	plt.title("Ritz Values Vs. No. of iterations")
	plt.scatter(rvals[:, 1], rvals[:, 0])
	plt.savefig("problem_6c.png")
	plt.show()

if __name__ == '__main__':
	main()