#!/usr/bin/env python
# coding: utf-8

# # Simulation
# 
# In this page we will simulate 100 individuals and create the discrete probabilities. I will plot some graph using the data. 
# 
# ```{warning} 
# Remember that `dhr.func` is the sintax that calls the file with the functions see before.
# ```
# 
# ## Packages

# In[1]:


import numpy as np
import pandas as pd
import dhr_functions as dhr


# ## Using the model solution
# Before simulating the data and the probabilities, I will run the model solution

# In[2]:


probs_w, _, _, _, _, vf = dhr.model_solution()


# ## Simulation
# ### Parameters
# We need some parameters to run our simulation:
#   - Number of simulation 
#   - μ (value of leisure)
#   - Non pecuniary cost
#   - Total days in the month
#   - Shock mean
#   - Shock standart deviation
#   - Seed to save the random results

# In[3]:


sim_nb   = 100
mu       = 4
P        = 3
Tm       = 30
eps_mean = 0 
eps_sd   = 1
np.random.seed(1996)


# ### Creating Matrices
# Now I will create the matrices to store the simulation results. Each colum stores the individual simulated data for all the years

# In[4]:


# Store leisure decision
L_dec = np.empty((Tm,sim_nb))
L_dec[:] = np.nan         

# Store days worked until t 
d_dec = np.empty((Tm,sim_nb))
d_dec[:] = np.nan         

# Store days worked in t + 1
d1_dec = np.empty((Tm,sim_nb))
d1_dec[:] = np.nan         

# Store ϵ received in t 
e_dec = np.empty((Tm,sim_nb))
e_dec[:] = np.nan         

# Utility gain from optimal choice
u_dec = np.empty((Tm,sim_nb))
u_dec[:] = np.nan         


# ### Simulating the individuals
# We will simulated our indiviuals follow these steps:
# 1. Draw the shock
# 2. Compute the value for working and leisure
# 3. Pick the max
# 4. Store the data

# In[5]:


for s in range(sim_nb):
    
    # Every individual starts with no days worked
    d = 0
    
    for t in range(30):
        
        # Store days worked until t
        d_dec[t,s] = d
        
        # Need condition for last period
        if t != 29:
            
            # 1. Draw shock
            e_shock = np.random.normal(loc = eps_mean, scale = eps_sd)
            
            # 2. Compute values for decisions: Using Value function from model solution
            w = vf[t+1,d+1]
            l = e_shock + mu - P + vf[t+1,d]
            
            # 3. Pick optmal choice
            dec = max(w,l)
            u = dec
            if dec == w:
                dec = 0
            else:
                dec = 1
            if dec == 0:
                d += 1
            else:
                pass
        else:
            
            # 1. Draw shock
            e_shock = np.random.normal(eps_mean, eps_sd)
            
            # 2. Compute values for decisions
            w = dhr.income_g(t, d+1)
            l = e_shock + dhr.income_g(t, d)
            
            # 3. Pick optmal choice
            dec = max(w,l)
            u = dec
            if dec == w:
                dec = 0 
            else:
                dec = 1
            if dec == 0:
                d += 1
            else:
                pass
                    
        # 4. Store results
        L_dec[t,s] = dec
        d1_dec[t,s] = d
        e_dec[t,s] = e_shock
        u_dec[t,s] = u


# Now we will create a dataframe using the simulated matrices

# In[6]:


simulated_data = []
for i in range(sim_nb):
    s = {}
    s['id'] = i
    s['t']  = list(range(1,31))
    s['d+1']  = list(d1_dec[:,i])
    s['d']    = list(d_dec[:,i])
    s['L']  = list(L_dec[:,i])
    s['eps']  = list(e_dec[:,i])
    s['U']  = list(u_dec[:,i])
    s = pd.DataFrame(s)
    simulated_data.append(s)
    
simulated_data = pd.concat(simulated_data, axis = 0).reset_index().drop('index', axis = 1)


# ## Discrete probabilities
# Using the model solution, we can create the discrete probabilities matrix for the
# days worked. First, create a matrix to store the results.

# In[7]:


prob_sim = np.zeros((Tm,Tm))


# We will start with every individual in the same position, wich means a mass of 1 in t = 0 and d = 0

# In[8]:


prob_sim[0,0] = 1 #P(t = 1, d = 0)


# To compute the discrete probability matrix, we will use two things:
# - The probability of working in t conditional on d (from the model solution)
# - The mass of teachers in t-1 and with an specific value of d

# In[9]:


for t in range(1,Tm):
    for d in range(t):
        prob_mass = prob_sim[t-1,d] # Mass of teacher in t = t-1, d = d
        prob_w_t_d = probs_w[t-1,d] # P(L = 0 | t = t-1, d = d)
        if np.isnan(prob_w_t_d):    # Maybe is not possible, but I defined as nan in the model solution
            prob_w_t_d = 0
        else:
            pass
        prob_sim[t,d]   = prob_sim[t,d] + (1-prob_w_t_d)*prob_mass # Adding the 0 to the teachers proportion multiplied by the conditional prob to leisure
        prob_sim[t,d+1] = prob_sim[t,d+1] + prob_w_t_d*prob_mass   # Same as (214) but now with prob to work


# ## Functions
# The simulation and the discrete probabilities are also included in the functions file

# In[10]:


_, L_dec, d_dec, d1_dec, e_dec, u_dec, _= dhr.simulate_data()
L_dec


# The probability function

# In[11]:


prob_sim = dhr.discrete_probs()
prob_sim


# ## Plots
# ### Plotting histories simulated

# In[12]:


dhr.simulation_plot()


# ```{figure} ../../images/dhr/simulation_histories_5_sim_control.png
# ```
# 
# ### Plotting probability of working, probability of days worked and EVs

# In[13]:


dhr.t_graphs()


# ```{figure} ../../images/dhr/EVs.png
# ```
# ```{figure} ../../images/dhr/prob_days_worked.png
# ```
# ```{figure} ../../images/dhr/prob_working.png
# ```
