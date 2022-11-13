---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Estimation

This page estimates the model parameters. We will do this in two steps:
1. [Using a grid of parameters values](grid)
2. [Using a Optimizer](opt)

```{warning} 
Remember that `dhr.func` is the sintax that calls the file with the functions see before.
```

## Packages

```{code-cell} ipython3
import time
import numpy as np
import pandas as pd
import seaborn as sns
import dhr_functions as dhr
from scipy.optimize import minimize
from matplotlib import pyplot as plt
```

## Calling the simulated data for the treatment

First we will call our simulated data using the function and transform in a dictionary. The dictionary makes 
the estimation faster. The dataframe_simulation function creates two dataset (control and treatment) and put them 
in a dictionary to return. `dataset` is going to store this dictionary.

```{code-cell} ipython3

dataset = dhr.dataframe_simulation()
dataset['treatment']
```

Select the group you want and take it from the dictionary. We will use the treatment.

```{code-cell} ipython3
data = dataset['treatment'].to_dict('records')
```
## Example with initial values
Now, set the initial values for the estimation, the list is [beta, theta]. After determining this, call the 
model solution to extract the probabilities of working and leisure.

```{code-cell} ipython3
theta=[0.01,4.5]
probs_w, probs_l, _, _, _, _, = dhr.model_solution(beta = theta[0], mu = theta[1]);
```

Now we will calculate the Maximum Likelihood for a specific initial value. This will be repeated multiple times
in the estimation in order to find the minimu log likelihood value.

```{code-cell} ipython3
logL = 0
for i in range(len(data)):
    t = int(data[i]['t'])
    d = int(data[i]['d'])
    L = int(data[i]['L'])
    likelihood = np.log(probs_w[t-1,d])*(1-L)+ np.log(probs_l[t-1,d])*L
    logL = logL + likelihood 
```

Now, check the value

```{code-cell} ipython3
-logL
```

Our estimation wants to minimize this, in order to match the model solution and the simulated data. Let's see this in two ways:
- Using a grid of parametes values
- Optmizer

(grid)=

## Grid of values

We will create a grid of values for beta and mu. After that, we will compute the Likelihood of the combinations and pick
the combination that generates the lowest value.

```{code-cell} ipython3
n = 10
b_guess  = np.linspace(0.01, 0.04,n)
mu_guess = np.linspace(3, 4.5,n)
```

Above I create a grid with 10 values each for beta and mu. Let's compute the Likelihoods. I will use a function called
Log_likelihood(), wich computes and stores parameters combinations and repective likelihood values in a dataframe.

```{code-cell} ipython3
:tags: ["hide-output"]
start = time.perf_counter()
log_values = dhr.Log_likelihood(data, b_guess, mu_guess)
maxL = log_values['likelihood'].max()
optb = round(log_values.loc[ log_values['likelihood'] == maxL]['beta'].item(),2)
optmu = round(log_values.loc[ log_values['likelihood'] == maxL]['mu'].item(),2)
end = time.perf_counter()
```
Let's see the result

```{code-cell} ipython3
print('optimun beta:' + str(optb) +'\noptimun mu:' + str(optmu))
print("Elapsed (with compilation) = {}s".format((end - start)))
```

The result is exaclty the parameters estimated in the model. Now, lets plot this and see graphically.

```{code-cell} ipython3
:tag: ['hide-cell']
# PLOTTING SETTINGS
plt.rcParams['figure.dpi'] = 500
plt.rcParams['savefig.dpi'] = 500
sns.set_style('ticks')
sns.despine(left=False, bottom=True)
sns.set_context("paper")
```

Above I just defined settings to improve plotting quality

```{code-cell} ipython3
sns.lineplot(data = log_values, x = 'mu', y = (log_values['likelihood']*-1), hue = 'beta', palette= 'nipy_spectral_r' ).set(title = 'Estimation using Likelihood Minimization')    
sns.despine(left=False, bottom=False)
```
(opt)=

## Optimizer

Now, use a minimizer (here `minimize` from scipy package) to find the parameters values that minimizes the log likelihood.
We need to define the function used to estimate the logLikelihood and put this as an argument of the function (`opt_likelihood`).
Also, we need to choose the initial guess and method.

```{code-cell} ipython3
:tags: ["hide-output"]
def opt_likelihood(theta):
    probs_w, probs_l, _, _, _, _, = dhr.model_solution(beta = theta[0], mu = theta[1]);
    logL = 0
    for i in range(len(data)):
        t = int(data[i]['t'])
        d = int(data[i]['d'])
        L = int(data[i]['L'])
        likelihood = np.log(probs_w[t-1,d])*(1-L)+ np.log(probs_l[t-1,d])*L
        logL = logL + likelihood 
    return -logL

start = time.perf_counter()
resu = minimize(opt_likelihood, theta, method = "SLSQP" )
end = time.perf_counter()
```

Now, print the results and see that the values that minimize the function are close to beta = 0.03 and mu = 4, the estimated
values from the paper.

```{code-cell} ipython3
print('Optimun beta:' + str(round(resu.x[0],2)) + "\n" + 'Optimun mu:' + str(round(resu.x[1],2)))
print("Elapsed (with compilation) = {}s".format((end - start)))
```