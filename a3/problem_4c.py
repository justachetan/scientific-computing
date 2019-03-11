import numpy as np
from problem_4a import make_data
from problem_4b import pca

def main():
	data = make_data()
	pcs, u, s, vh, Y = pca(data.copy())
	mean = np.mean(data, axis=1)
	Y_n = np.dot(np.dot(u, np.diag(s)), vh)
	rel_err = np.linalg.norm(Y_n - Y, ord=2) / np.linalg.norm(Y, ord=2)
	print("Relative error of Y:", rel_err)

if __name__ == '__main__':
	main()