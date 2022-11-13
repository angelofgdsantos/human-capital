'''

Packages

'''
import time
import numpy as np
import pandas as pd
import seaborn as sns
import dhr_functions as dhr
from matplotlib import pyplot as plt
from scipy.optimize import minimize
'''

Simulate data for treatment and control to get the momments (means) using the estimated parameters (beta = 0.03 and mu = 4)

'''
data = dhr.dataframe_simulation()
data_t = data['treatment']
data_c = data['control']
d_mean_t0 = data_t.loc[data_t['t'] == 30]['d+1'].mean()
d_mean_c0 = data_c.loc[data_c['t'] == 30]['d+1'].mean()
'''

The means are

'''
print(d_mean_c0) # control
print(d_mean_t0) # treatment
'''

Let's check a different value. With these values, the roots are very different.

'''
theta = [0.03,0]
dhr.root_calibration(theta)
'''

Now we will use the optmizer again to find the combination of parameters value that match the momments of our data
using the estimated parameters. Again, we neet to define the function to use the minimizer. 

'''
def root_calibration(theta):
    d_mean_c0 = 5.2
    d_mean_t0 = 20.82
    means, _, _, _, _, _, _, = dhr.simulate_data(theta, M = 10, I = 50, gs = 1)
    print(means)
    f = ( means[0,1] - d_mean_c0)**2 + (means[0,0] - d_mean_t0)**2
    return f

start = time.perf_counter()
theta = [0.01,4]
resu = minimize(root_calibration,theta, method='Nelder-Mead')
end = time.perf_counter()

'''

Let's seet the roots

'''
print('Optimun beta:' + str(round(resu.x[0],2)) + "\n" + 'Optimun mu:' + str(round(resu.x[1],2)))
print("Elapsed (with compilation) = {}s".format((end - start)))
'''

This matches exaclty the moments that we see in the data

'''