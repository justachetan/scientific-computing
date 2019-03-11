from __future__ import division
import numpy as np
import scipy as sp

def gen_vectors():

    b_all = np.random.rand(110, 512)
    perturbations = np.arange(0.01, 0.11, 0.01)

    for i in range(10):
        b_all[i] = b_all[i] / np.linalg.norm(b_all[i], ord=2)
    
    for i in range(10, b_all.shape[0], 10):

        for j in range(10):
            # print(np.linalg.norm(b_all[i//10 - 1], ord=2), np.linalg.norm((b_all[i + j] / np.linalg.norm(b_all[i + j], ord=2)) * perturbations[i//10 - 1], ord=2))
            
            b_all[i + j] = b_all[i//10 - 1] + ((b_all[i + j] / np.linalg.norm(b_all[i + j], ord=2)) * perturbations[i//10 - 1])


    return b_all


if __name__ == '__main__':
    b_all = gen_vectors()
    print(b_all.shape)
