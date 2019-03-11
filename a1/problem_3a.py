

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.special
import pandas as pd


# # Part (a)

# Code might send out RunTime warnings. These are benign, caused due to Python 2.7.
# Can verify here: https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility


def get_approx_jv(n, z):
    
    # returns approximate value of J_{n+1}
    
    if (n < 1).any() == True:
        return ValueError
    
    return ( ( ( ( 2 * n ) / z ) * scipy.special.jv(n, z) ) - scipy.special.jv(n - 1, z) )





z = 20



true_jv = scipy.special.jv(np.arange(2, 51), z)
approx_jv = get_approx_jv(np.arange(1, 50), z)





# I have printed the magnitude of the Relative Error

relative_err = (approx_jv - true_jv) / true_jv
relative_err = np.abs(relative_err)





df = pd.DataFrame({"n" : [i for i in range(2, 51)], "LHS" : true_jv, "RHS" : approx_jv, "Relative Error" : relative_err}, columns=["n", "LHS", "RHS", "Relative Error"])
print(df.to_string(index_names=False))









plt.xlabel("Value of n")
plt.ylabel("Absolute Value of Relative Error")
plt.title("Problem 3(a)")
plt.plot( np.arange(2, 51), relative_err)
plt.savefig("problem_3a.png")

# Uncomment to see figures
plt.show()

