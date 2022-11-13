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

# Model solution

In this exercise, we will go through the model solution.

```{warning} 
Remember that `dhr.func` is the sintax that calls the file with the functions see before.
```

## Packages

```{code-cell} ipython3
import numpy as np
import dhr_functions as dhr
from scipy.stats import norm
```

## Parameters

Define parameters of the model. The parameters are:
- Group we are simulating 
- Discount factor
- Total days in the month
- Value for utility of leisure
- Non pecuniary cost of working
     
```{code-cell} ipython3
group = "treatment"
beta  = 0.03
Tm    = 30
mu    = 4
P     = 3
```

## Creating matrices 

First we will create the matrices that we will fill using the model solution.
We will store the following: 
- Probability of working
- Probability of leisure
- Expected value of working
- Expected value of leisure
- ϵ Threshold
- Value function 
    
```{code-cell} ipython3
# Probability matrix
probs_w = np.empty((Tm,Tm)) 
probs_w[:] = np.nan         

probs_l = np.empty((Tm,Tm)) 
probs_l[:] = np.nan         

# Threshold matrix
eps_t = np.empty((Tm,Tm)) 
eps_t[:] = np.nan         

# Expected values matrix for work
eve_w  = np.empty((Tm,Tm)) 
eve_w[:] = np.nan           

# Expected values matrix for leisure
eve_l  = np.empty((Tm,Tm)) 
eve_l[:] = np.nan           

# Value function matrix
vf = np.empty((Tm,Tm)) 
vf[:] = np.nan  
```
## Model solution loop

The model solution is solved backward and follows these steps:

1. Solve for the threshold that makes the value function for leisure bigger.
2. Use the result from 1 to obtain the probability of leisure and work.
3. Calculate the truncated distribution value for leisure
4. Calculate the EV (expected values in the period ahead)
5. Repeat until the end of the loop

This gives the value of having an specific level of education in the last period

```{code-cell} ipython3
:tags: ["hide-output"]
for t in range(Tm,-1,-1):
    print()
    for d in range(0,Tm):
        if d >= t:
            pass
        else:            
            if t == 30:
                eps_t[t-1,d] = beta*(dhr.income_g(t, d+1)- dhr.income_g(t, d))- (mu-P)
            else:
                eps_t[t-1,d] = vf[t,d+1] - vf[t,d] - (mu-P)
            probs_l[t-1,d] = 1 - norm.cdf(eps_t[t-1,d])
            probs_w[t-1,d] = 1 - probs_l[t-1,d]
            
            if t == 30:
                eve_l[t-1,d] = 0 + dhr.exp_epsilon(eps_thres = eps_t[t-1,d])
                eve_w[t-1,d] = 0
            else:
                eve_l[t-1,d] = vf[t,d] + dhr.exp_epsilon(eps_thres = eps_t[t-1,d])
                eve_w[t-1,d] = vf[t,d+1]
            
            vf[t-1,d] = probs_l[t-1,d]*(dhr.VF(1, t, d, eve_l = eve_l[t-1,d])) + \
            probs_w[t-1,d]*(dhr.VF(0, t, d+1, eve_w = eve_w[t-1,d])) 
```

Let's see the value function matrix

```{code-cell} ipython
:tags: ["hide-output"]
print(vf)
```

Now, I will run the above loop using the model solution function, which will create the same matrices 

```{code-cell} ipython3
:tags: ["hide-output"]
probs_w, probs_l, eps_t, eve_w, eve_l, vf = dhr.model_solution()
print(vf)
```