import numpy as np
import matplotlib.pyplot as plt
from problem_2a import QR_fact_iter

def back_subsitute(U, bb):
	n = U.shape[1]
	x = np.zeros(n)
	for j in range(n - 1, -1, -1):   # loop backwards over columns
		if U[j, j] == 0:
			raise RuntimeError("singular matrix")

		x[j] = bb[j] / U[j, j]
		for i in range(0, j):
			bb[i] -= U[i, j] * x[j]

	return x



def fit_poly_on_data(data, order=1):
	A = np.array([[float(i) ** j for j in range(order+1)] for i in range(1, len(data) + 1)])
#     print(A)
	b = data
	Q, R = QR_fact_iter(A)
	d = np.dot(Q.T, b)
	c = back_subsitute(R, d)
	return c

def str_poly(coeff):
	if len(coeff) > 6:
		raise RuntimeError("Degree > 5. Please handle.")
	
	template = {0 : "", 1 : "x", 2 : "x^2", 3 : "x^3", 4 : "x^4", 5 : "x^5",}
	letters = ["a", "b", "c", "d", "e", "f"]
	
	string = ""
	values = "\n\n"
	
	for i in range(len(coeff), 1, -1):
		
		string = string + letters[len(coeff) - i] + " * " + template[i - 1] + " + "
		values = values + letters[len(coeff) - i] + " = " + str(coeff[i - 1]) + "\n"
		
	string = string + letters[len(coeff) - 1]
	values = values + letters[len(coeff) - 1] + " = " + str(coeff[0])
	string = string + values
	
	
	return string


def get_relative_residual(A, x, b):
	return np.linalg.norm(np.dot(A, x) - b, ord=2) / np.linalg.norm(b, ord=2)


def main():
	v = np.loadtxt("petrol_price_delhi.txt", delimiter="\n")
	Y_cal = list()
	Y_true = list()

	for i in range(1, 6):
		print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
		print("Results for Degree:", i)
		coeff_cal = fit_poly_on_data(v, order=i)
		A = np.array([[float(j) ** k for k in range(i+1)] for j in range(1, len(v) + 1)])
		coeff_true = np.linalg.lstsq(A, v, rcond=None)[0]
		rel_res_cal = get_relative_residual(A, coeff_cal, v)
		rel_res_true = get_relative_residual(A, coeff_true, v)
		str_poly_cal = str_poly(coeff_cal)
		str_ploy_true = str_poly(coeff_true)
		print("\n\n")
		print("------------------------------------------------------------------------")
		print("Strategy: np.linalg.lstsq")
		print("------------------------------------------------------------------------")
		print("Approximate Polynomial:\n\n")
		print(str_ploy_true)
		print("\n\n")
		print("Relative Residual:", rel_res_true)
		print("\n\n")
		print("------------------------------------------------------------------------")
		print("Strategy: get_QR()")
		print("------------------------------------------------------------------------")
		print("Approximate Polynomial:\n\n")
		print(str_poly_cal)
		print("\n\n")
		print("Relative Residual:", rel_res_cal)
		print("\n\n")
		Y_cal.append(np.dot(A, coeff_cal))
		Y_true.append(np.dot(A, coeff_true))

	plt.figure(figsize=(10, 10))
	plt.subplots_adjust(hspace=8)
	plt.subplot(2, 1, 1)
	plt.scatter(np.arange(1, len(v) + 1), v, marker="+", c="black", label="Original Data")
	# plt.plot(np.arange(1, len(v) + 1), Y_true[0], np.arange(1, len(v) + 1), Y_true[1], np.arange(1, len(v) + 1), Y_true[2], np.arange(1, len(v) + 1), Y_true[3], np.arange(1, len(v) + 1), Y_true[4])
	plt.plot(np.arange(1, len(v) + 1), Y_true[0], label="Degree = 1")
	plt.plot(np.arange(1, len(v) + 1), Y_true[1], label="Degree = 2")
	plt.plot(np.arange(1, len(v) + 1), Y_true[2], label="Degree = 3")
	plt.plot(np.arange(1, len(v) + 1), Y_true[3], label="Degree = 4")
	plt.plot(np.arange(1, len(v) + 1), Y_true[4], label="Degree = 5")
	plt.xlabel("Day")
	plt.ylabel("Price of petrol (Rs./l)")
	plt.title("Data fitted using np.linalg.lstsq()")
	plt.legend(loc=1)
	plt.subplot(2,1,2)
	plt.scatter(np.arange(1, len(v) + 1), v, marker="+", c="black", label="Original Data")
	plt.plot(np.arange(1, len(v) + 1), Y_cal[0], label="Degree = 1")
	plt.plot(np.arange(1, len(v) + 1), Y_cal[1], label="Degree = 2")
	plt.plot(np.arange(1, len(v) + 1), Y_cal[2], label="Degree = 3")
	plt.plot(np.arange(1, len(v) + 1), Y_cal[3], label="Degree = 4")
	plt.plot(np.arange(1, len(v) + 1), Y_cal[4], label="Degree = 5")
	plt.xlabel("Day")
	plt.ylabel("Price of petrol (Rs./l)")
	plt.title("Data fitted using get_QR()")
	plt.legend(loc=1)
	plt.tight_layout()
	plt.savefig("problem_2c.png")
	plt.show()

if __name__ == '__main__':
	main()