---
jupytext:
  cell\_metadata\_filter: -all
  formats: md:myst
  text\_representation:
extension: .md
format_name: myst
format_version: 0.13
jupytext_version: 1.11.5
kernelspec:
  display\_name: Python 3
  language: python
  name: python3
---

# Model solution

In this exercise, we will go through the model solution.


```{warning} 
Remember that `ams.func` is the sintax that calls the file with the functions see before.
```


## Packages

First, calls numpy and my functions file so we can use it in our model solution.


```{code-block} ipython3
import numpy as np
import ams_functions as ams
```

## Parameters

To start our solution, we need to determine the parameters. The names here can be a little bit confusion, but the `age_max` indicate how many adding yeas the child can study. This means that we are simulated children that are 6 years old and can study until they are 16. 

```{code-block} ipython3
age_start = 6
age_max   = 10
```

Before the model solution loop, we need to create empty matrices to store our results. I commented the code so you can see what is the matrix about.

```{code-block} ipython3
# Probabilities of working
probs_w = np.empty((age_max,age_max)) 
probs_w[:] = np.nan         

# Probabilities of school
probs_s = np.empty((age_max,age_max)) 
probs_s[:] = np.nan   

# Threshold matrix
eps_t = np.empty((age_max,age_max)) 
eps_t[:] = np.nan         

# Expected values matrix for L = 1 
eve_w  = np.empty((age_max,age_max)) 
eve_w[:] = np.nan              

# Expected values matrix for L = 0 
eve_s = np.empty((age_max,age_max)) 
eve_s[:] = np.nan   

# Value function matrix
vf = np.empty((age_max,age_max)) 
vf[:] = np.nan  
```

## Model solution loop

The model solution is solved backward and follows these steps:

1. Solve for the threshold that makes the value function for going to school bigger.
2. Use the result from 1 to obtain the probability of going to school.
3. Calculate the truncated distribution value for going to school
4. Calculate the EV (expected values in the period ahead)
5. Repeat until the end of the loop

In this case, for the last period we will use the terminal value defined as:

\begin{align}

V(ed_{i,18}) = \frac{\alpha_{1}}{1 + exp(-\alpha_{2}*ed_{i,18})}

\end{align}

This gives the value of having an specific level of education in the last period

```{code-block} ipython3

for age in range(age_max,-1,-1):
    for edu in range(0,age_max):
        if edu >= age:
            pass
        else:       
            # For the threshold we will use the terminal value to capture the value of an specifiy education level in terms of utility  
               
            if age == age_max: #last period situation
                tv_s =  ams.prob_progress(edu)*(ams.terminal_v(edu = edu+1)) + (1 - ams.prob_progress(edu))*(ams.terminal_v(edu = edu)) 
                tv_w =  ams.terminal_v(edu = edu)
                eps_t[age-1,edu] = ams.value_function(age,edu,1, EV = tv_s) - ams.value_function(age,edu,0, EV = tv_w)
                                   
            else:                #Other periods will take the difference between value functions
                tv_s = ams.prob_progress(edu)*(vf[age, edu+1]) + (1 - ams.prob_progress(edu))*(vf[age,edu])
                tv_w = vf[age, edu]
                eps_t[age-1,edu] = ams.value_function(age,edu,1, EV = tv_s) - ams.value_function(age,edu,0, EV = tv_w)
            
            # Now we will calcute the probabilities of schooling and working based on the thresholds                     
            probs_s[age-1,edu] = ams.logistic(eps = eps_t[age-1,edu])
            probs_w[age-1,edu] = 1 - probs_s[age-1,edu]
            
            # Using the thresholds we will calculate the EVs
            if age == age_max:
                u_t = ams.trunc_change(eps = eps_t[age-1,edu])
                eve_s[age-1,edu] = ams.value_function(age,edu,1, EV = tv_s) + ams.trunc_school(ϵ_threshold = eps_t[age-1,edu], u_threshold = u_t)
                eve_w[age-1,edu] = ams.value_function(age,edu,0, EV = tv_w)
            else:
                u_t = ams.trunc_change(eps = eps_t[age-1,edu])
                eve_s[age-1,edu] = ams.value_function(age,edu,1, EV = tv_s) + ams.trunc_school(ϵ_threshold = eps_t[age-1,edu], u_threshold = u_t)
                eve_w[age-1,edu] = ams.value_function(age,edu,0, EV = tv_w)
            
            vf[age-1,edu] = probs_s[age-1,edu]*eve_s[age-1,edu] + probs_w[age-1,edu]*eve_w[age-1,edu]
```

Let's some results. The probability of going to school matrix looks like this:

```{code-block} ipython3
probs_s
```

The EV matrix is:

```{code-block} ipython3
vf
```

Using our function file, you can use the following code to run the loop we just saw

```{code-block} ipython3
probs_w, probs_s, eps_t, eve_w, eve_s, vf = ams.ams_solution()
```