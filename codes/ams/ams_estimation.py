'''

Packages

'''
import time
import numpy as np
import pandas as pd
import seaborn as sns
import ams_functions as ams
from scipy.optimize import minimize
'''
We will estimate these for three parameters:
 - μ: Shifter coefficient for the child
 - g: Coefficient in the utility for the grant
 - b_mom: Mother background coefficient
 
First we will call our simulated data using the function and transform in a dictionary. The dictionary makes 
the estimation faster. The dataframe_simulation function creates two dataset (control and treatment) and put them 
in a dictionary to return. `dataset` is going to store this dictionary.

'''
data = ams.simulation()
data = data.to_dict('records')
'''

Now, set the initial values for the estimation, the list is [beta, theta]. After determining this, call the 
model solution to extract the probabilities of working and leisure.

'''
g = 3
μ = -8
b_mom = -0.5
theta = np.array([g, μ, b_mom])
probs_w, probs_s, _, _, _, _, = ams.solution(theta=theta)
'''

Now we will calculate the Maximum Likelihood for a specific initial value. This will be repeated multiple times
in the estimation in order to find the minimu log likelihood value.

'''
logL = 0
for i in range(len(data)):
    age = int(data[i]['age'])
    edu = int(data[i]['edu'])
    S = int(data[i]['School'])
    likelihood = np.log(probs_w[age-6,edu])*(1-S)+ np.log(probs_s[age-6,edu])*S
    logL = logL + likelihood 
'''

Now, check the value

'''
-logL
'''

Our estimation wants to minimize this, in order to match the model solution and the simulated data. Lets see the 
optmizer. Let's see this in two ways:
- Using a grid of parametes values
- Optmizer

## Grid of values

We will create a grid of values for g, μ and b_mom. After that, we will compute the Likelihood of the combinations and pick
the combination that generates the lowest value.
'''
n = 20
g_guess  = np.linspace(2, 3.5,n)
#μ_guess = np.linspace(1, -1,n)
#b_mom_guess = np.linspace(-6, -10,n)
'''

Above I create a grid with 10 values each for beta and mu. Let's compute the Likelihoods. I will use a function called
Log_likelihood(), wich computes and stores parameters combinations and repective likelihood values in a dataframe.

'''
start = time.perf_counter()
log_values = ams.Log_likelihood(data, g_guess)
maxL = log_values['likelihood'].max()
optg = round(log_values.loc[ log_values['likelihood'] == maxL]['gamma'].item(),2)
optmu = round(log_values.loc[ log_values['likelihood'] == maxL]['mu'].item(),2)
optmom = round(log_values.loc[ log_values['likelihood'] == maxL]['mom'].item(),2)
print('optimun gamma:' + str(optg) +'\n optimun mu:' + str(optmu) + '\n optimun mu:' + str(optmom))
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))
