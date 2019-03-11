

from __future__ import division
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.special



# Code may print some warnings due to pandas in Python2. 
# These warning are benign. Verigy here: https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility



z = 20




J_0 = scipy.special.jv(0, z)
J_1 = scipy.special.jv(1, z)





bessel_val_by_rec = [J_0, J_1]





for i in range(2, 51):
    
    J_i = ( ( ( ( 2 * ( i - 1 ) ) / z ) * bessel_val_by_rec[i - 1] ) - bessel_val_by_rec[i - 2] )

    bessel_val_by_rec.append(J_i)





bessel_val_by_func = [scipy.special.jv(i, z) for i in range(0, 51)]





relative_err = []
for i in range(2, 51):
    relative_err.append((bessel_val_by_rec[i] - bessel_val_by_func[i])/bessel_val_by_func[i])

# Again, just printing magnitude of the Relative Error
# Comment this value to print just the ratio
relative_err = [abs(i) for i in relative_err]





df = pd.DataFrame({"n" : [i for i in range(2, 51)], "From Function" : bessel_val_by_func[2:], "From Recurrence" : bessel_val_by_rec[2:], "Relative Error" : relative_err}, columns=["n", "From Function", "From Recurrence", "Relative Error"])
print(df.to_string(index_names=False))







plt.xlabel("Value of n")
plt.ylabel("Absolute Value of Relative Error")
plt.plot( np.arange(2, 51), relative_err)
plt.savefig("problem_3b.png")
# Uncomment to show plot
# plt.show()

