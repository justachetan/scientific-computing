from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from problem_6a import construct_A
from problem_6b import gen_vectors
from matplotlib import rc



def generate_four_tuples(A, b_all):

	tuples = []

	for i in range(10):

		b = b_all[i]

		x = np.linalg.solve(A, b)

		for j in range(10):
			# print(i, j)
			b_db = b_all[(i+1) * 10 + j]
			x_dx = np.linalg.solve(A, b_db)

			db = b_db - b
			dx = x_dx - x

			tuples.append((b, db, x, dx))

	return np.array(tuples)

def generate_plot(cno_A, tuples):

	points_x = []
	points_y = []


	for i in range(len(tuples)):

		db_norm = np.linalg.norm(tuples[i][1], ord=2)
		b_norm = np.linalg.norm(tuples[i][0], ord=2)
		dx_norm = np.linalg.norm(tuples[i][3], ord=2)
		x_norm = np.linalg.norm(tuples[i][2], ord=2)

		x = cno_A * (db_norm / b_norm)
		y = dx_norm / x_norm

		points_x.append(x)
		points_y.append(y)
	
	points_x = np.array(points_x, dtype=np.float128)
	points_y = np.array(points_y, dtype=np.float128)

	plt.figure()
	plt.xlabel(u"cond(A) * (||\delta b|| / ||b||)")
	plt.ylabel(u"||\delta x|| / ||x||")
	plt.plot(points_x, points_y, "ko")
	plt.savefig("problem_6c.png")
	# Uncomment this line to see the plot
	# plt.show()

def main():

	A, cno_A = construct_A(512)
	b_all = gen_vectors()
	tuples = generate_four_tuples(A, b_all)
	generate_plot(cno_A, tuples)

if __name__ == '__main__':
	main()


