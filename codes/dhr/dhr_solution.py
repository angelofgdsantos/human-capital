"""

Author: Angelo Santos

This code solves the DHR model 

                                              DHR PAPER - TEACHERS ABSENTEEISM
MODEL:
    STATE VARS:  
        - t: day of month
        - d: days worked until t
        - eps_tm: shock received in t on month m
    CHOICE VARS: 
        - L: Leisure or work
    PARAMETERS: 
        - Tm: total days of a month
        - mu_bar: mu - P
            * mu: value of leisure
            * P: non-pecuniary cost of not working. This will be set as 0
        - Prob_fire: set as 0, there is no firing in data
        - Beta: set as 1

    CONTROL:  
        - For t < Tm                                                         [(mu_bar + eps_tm) + EVF(t+1,d,eps_t+1,m)]  if L = 1
            * VF(t,d,eps_tm): max(L) Prob_fire(t,d)*F + (1 - Prob_fire(t,d))*                   
                                                                             [EVF(t+1,d+(1-L),eps_t+1,m)]                if L =0
        
        - For t = Tm                                                          [mu_bar + eps_tm) + EVF(t+1,d,eps_t+1,m) + beta*1000]
            * VF(Tm,d,esp_tm): max(L) Prob_fire(Tm,d)* F + (1-Prob_fire(Tm,d))*
                                                                              [beta*1000 + EVF(1,0,esps_1,m+1)]
    TREATMENT:  
        - For t < Tm                                                         [(mu_bar + eps_tm) + EVF(t+1,d,eps_t+1,m)]  if L = 1
            * VF(t,d,eps_tm): max(L) Prob_fire(t,d)*F + (1 - Prob_fire(t,d))*                   
                                                                             [EVF(t+1,d+(1-L),eps_t+1,m)]                if L =0
        
        - For t = Tm                                                          [mu_bar + eps_tm) + EVF(1,0,eps_t,m+1) + beta*{ 500 + MAX{0,d-10}*50}
            * VF(Tm,d,esp_tm): max(L) Prob_fire(Tm,d)* F + (1-Prob_fire(Tm,d))*
                                                                              [beta*{ 500 + MAX{0,d+1-10}*50} + EVF(1,0,esps_1,m+1)]
    PROBABILITIES:
        IF d-1 <= 10 ==> L(0,t,d) = Normal_cdf(mu)

"""
import numpy as np
import dhr_functions as dhr
from scipy.stats import norm
'''

Parameters
==========

Define parameters of the model. The parameters are:
    - Group we are simulating 
    - Discount factor
    - Total days in the month
    - Value for utility of leisure
    - Non pecuniary cost of working
    
'''
group = 'treatment'
beta  = 0.03
Tm    = 30
mu    = 4
P     = 3
'''

First we will create the matrices that we will fill using the model solution.
We will store the following: 
    - Probability of working
    - Probability of leisure
    - Expected value of working
    - Expected value of leisure
    - ϵ Threshold
    - Value function 
    
'''
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
'''

## Model solution loop

The model solution is solved backward and follows these steps:

1. Solve for the threshold that makes the value function for leisure bigger.
2. Use the result from 1 to obtain the probability of leisure and work.
3. Calculate the truncated distribution value for leisure
4. Calculate the EV (expected values in the period ahead)
5. Repeat until the end of the loop

This gives the value of having an specific level of education in the last period

'''    
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
'''

Let's see some results. The probability of leisure and value function matrix

'''            
print(probs_w)
print(vf)
'''

Now, I will run the above loop using the model solution function, which will create the same matrices 

'''
probs_w, probs_l, eps_t, eve_w, eve_l, vf = dhr.model_solution()