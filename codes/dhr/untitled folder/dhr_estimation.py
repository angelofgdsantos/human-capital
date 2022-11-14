
"""
Created on Sat Sep 24 23:56:55 2022

@author: angelosantos

THIS CODE ESTIMATES THE MODEL USING SIMULATED DATA:
    1 Use the model solution to simulate individual SS
    2 Guess parameters
    3 Solve for probs
    4 Solve for EVs
    5 Construct MLE
    6 Repeat above until convergence
    
"""
#%%
import os 
import time
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from matplotlib import pyplot as plt

############################################################################### PLOTTING SETTINGS
plt.rcParams['figure.dpi'] = 500
plt.rcParams['savefig.dpi'] = 500
sns.set_style('ticks')
sns.despine(left=False, bottom=True)
sns.set_context("paper")

############################################################################### FUNCTIONS

def Log_likelihood(data, b_guess, mu_guess, group = 'treatment'):
    log_values = []
    sim = 1
    for b in b_guess:
        for m in mu_guess:
            print('simulation '+ str(sim) + ' of ' + str(n*n))
            dic = {}
            probs_w, probs_l, eps_t, eve_w, eve_l, vf = model_solution(beta = b, mu = m, group = 'treatment');
            logL = 0
            for i in range(len(data)):
                t = int(data[i]['t'])
                d = int(data[i]['d'])
                L = int(data[i]['L'])
                likelihood = np.log(probs_w[t-1,d])*(1-L)+ np.log(probs_l[t-1,d])*L
                logL = logL + likelihood 
            dic['beta'] = b
            dic['mu'] = m
            dic['likelihood'] = logL
            log_values.append(dic)
            sim += 1
    return pd.DataFrame(log_values)


############################################################################### 1 - SIMULATED DATA
np.random.seed(1996)
sim_nb = 100
os.chdir('/Users/angelosantos/Documents/GitHub/human-capital/codes/dhr/untitled folder')
exec(open("DHR_model_solution_both.py").read())
data = pd.read_pickle('simulation_'+str(sim_nb)+'_treatment.pkl')
data = data.to_dict('records')

############################################################################### 2 - GUESS PARAMETERS

n = 10
b_guess  = np.linspace(0.01, 0.04,n)
mu_guess = np.linspace(3, 4.5,n)

start = time.perf_counter()
log_values = Log_likelihood(data, b_guess, mu_guess)
maxL = log_values['likelihood'].max()
optb = round(log_values.loc[ log_values['likelihood'] == maxL]['beta'].item(),2)
optmu = round(log_values.loc[ log_values['likelihood'] == maxL]['mu'].item(),2)
print('optimun beta:' + str(optb) +'\n optimun mu:' + str(optmu))
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))

############################################################################### 3 - PLOTTING

sns.lineplot(data = log_values, x = 'mu', y = (log_values['likelihood']*-1), hue = 'beta', palette= 'nipy_spectral_r' ).set(title = 'Estimation using Likelihood Minimization')    
sns.despine(left=False, bottom=False)
plotname = 'MLE.png'
plt.savefig('/Users/angelosantos/Documents/GitHub/human-capital/codes/dhr/plots/'+plotname)
plt.close()
# %%
