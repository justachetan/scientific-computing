

from __future__ import division
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.special




# Code may print some warnings due to pandas in Pyton 2.7
# It can be run in Python3 as well. The warnings are completely benign.



z = 20





J_50 = scipy.special.jv(50, z)
J_49 = scipy.special.jv(49, z)





bessel_val_by_rec = [J_50, J_49]





for i in range(48, -1, -1):
    
    J_i = ( ( ( ( 2 * ( i + 1 ) ) / z ) * bessel_val_by_rec[50 - i - 1] ) - bessel_val_by_rec[50 - i - 2] )

    bessel_val_by_rec.append(J_i)





bessel_val_by_func = [scipy.special.jv(i, z) for i in range(50, -1, -1)]





relative_err = []
for i in range(2, 51):
    relative_err.append((bessel_val_by_rec[i] - bessel_val_by_func[i])/bessel_val_by_func[i])

# Again, I am just printing magnitude of the Relative Error
# Comment this line to print just the ratio
relative_err = [abs(i) for i in relative_err]





df = pd.DataFrame({"n" : [i for i in range(2, 51)], "From Function" : bessel_val_by_func[2:], "From Recurrence" : bessel_val_by_rec[2:], "Relative Error" : relative_err}, columns=["n", "From Function", "From Recurrence", "Relative Error"])
print(df.to_string(index_names=False))







plt.xlabel("Value of n")
plt.ylabel("Absolute Value of Relative Error")
plt.title("Problem 3(d)")
plt.plot( np.arange(2, 51), relative_err)
plt.savefig("problem_3d.png")
# Uncomment to show plot
# plt.show()
