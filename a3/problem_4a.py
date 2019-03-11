import numpy as np
import matplotlib.pyplot as plt

def make_data(dims=2, npts=3000): 
    np.random.seed(13)
    mix_mat = np.random.randn(dims, dims) 
    mean = np.random.randn(dims)
    return np.dot( mix_mat, \
                  np.random.randn(dims, npts)) + mean[:, np.newaxis]

def main():
	data = make_data()
	plt.scatter(data[0], data[1])
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Plot of complete dataset")
	plt.savefig("problem_4a.png")
	plt.show()

if __name__ == '__main__':
	main()