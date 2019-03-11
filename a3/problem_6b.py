import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt





def lancoiz(A, max_iter=20):
    alphas = list()
    betas = list()
    ritz_vals = list()
    qk_1 = np.zeros(A.shape[1])
    qk = np.random.random(A.shape[1])
    qk = qk / np.linalg.norm(qk, ord=2)
    Q = list()
    Q.append(qk_1)
    Q.append(qk)
    beta = 0
    k = 1
    while True:
        qk = Q[k]
        qk_1 = Q[k-1]
        uk = np.dot(A, qk)
        ak = np.dot(qk, uk)
        alphas.append(ak)
        uk -= ((beta * qk_1) + (ak * qk))
        beta = npla.norm(uk, ord=2)
        if beta == 0:
            break
        betas.append(beta)
        qk_p1 = uk / beta
        
        T_i = np.diag(np.array(alphas))
        T_i = T_i + np.diag(np.array(betas[:len(alphas) - 1]), -1) + np.diag(np.array(betas[:len(alphas) - 1]), +1) 
        ritzs = np.linalg.eigvals(T_i)
        
        ritz_vals.extend([[k, ritzs[j]] for j in range(len(ritzs))])
        
        k+=1
        if k >= max_iter:
            break
        Q.append(qk_p1)

    H = np.diag(np.array(alphas))
    H = H + np.diag(np.array(betas[:len(alphas) - 1]), -1) + np.diag(np.array(betas[:len(alphas) - 1]), +1) 
    return np.array(Q[1:]).T, H, np.array(ritz_vals)
    


	




def main():

	B = np.random.random((100, 100))
	A = B + B.T
	Q, H, rvals = lancoiz(A, max_iter=20)




	part_1 = npla.norm(np.dot(Q, Q.T) - np.diag(np.ones(100)), ord = 2)





	part_2 = npla.norm(np.dot(np.dot(Q.T, A), Q) - H, ord=2) / npla.norm(A, ord=2)



	print("Answer for part 6(a) (i):", part_1)
	print("Answer for part 6(a) (ii):", part_2)


if __name__ == '__main__':
	main()



