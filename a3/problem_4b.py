import numpy as np
import matplotlib.pyplot as plt
from problem_4a import make_data

def pca(data):
    mean = np.mean(data, axis=1)
    data = np.array([data[i] - mean[i] for i in range(len(data))])
    N = data.shape[1]
    Y = (1 / np.sqrt(N - 1)) * data
    u, s, vh = np.linalg.svd(Y.copy(), full_matrices=False)
    sigma = np.diag(s)
    pc = np.dot(u, np.diag(s))
    return pc, u, s, vh, Y

def main():
	data = make_data()

	pcs, u, s, vh, Y = pca(data.copy())
	mean = np.mean(data, axis=1)
	plt.figure(figsize=(8, 4))
	plt.scatter(data[0], data[1])
	plt.gca().set_aspect("equal")
	plt.arrow(mean[0], mean[1], pcs[:, 0][0],  pcs[:, 0][1], head_width=0.05, head_length=0.1, fc='k', ec='k')
	plt.arrow(mean[0], mean[1], pcs[:, 1][0],  pcs[:, 1][1], head_width=0.05, head_length=0.1, fc='k', ec='k')
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Dataset with Principal Components")
	plt.savefig("problem_4b.png")
	plt.show()

if __name__ == '__main__':
	main()