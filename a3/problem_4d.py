import numpy as np
import matplotlib.pyplot as plt
from problem_4a import make_data
from problem_4b import pca

def main():
	data = make_data()
	pcs, u, s, vh, Y = pca(data.copy())
	s_d = s.copy()
	s_d[-1] = 0
	Y_d = np.dot(np.dot(u, np.diag(s_d)), vh)
	data_d = Y_d * np.sqrt(Y.shape[1] + 1) + np.mean(data, axis=1).reshape(Y_d.shape[0], 1)
	pcs_d, u_d, s_dd, vh_d, Y_d = pca(data_d.copy())
	mean_d = np.mean(data, axis=1)
	plt.figure(figsize=(8, 4))
	plt.scatter(data_d[0], data_d[1])
	plt.gca().set_aspect("equal")
	plt.arrow(mean_d[0], mean_d[1], pcs_d[:, 0][0],  pcs_d[:, 0][1], head_width=0.05, head_length=0.1, fc='k', ec='k')
	plt.arrow(mean_d[0], mean_d[1], pcs_d[:, 1][0],  pcs_d[:, 1][1], head_width=0.05, head_length=0.1, fc='k', ec='k')
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Reformed Dataset with Principal Components")
	plt.savefig("problem_4d.png")
	plt.show()

if __name__ == '__main__':
	main()