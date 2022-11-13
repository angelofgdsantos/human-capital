#%%[markdown]
# Importing packages
#%% 
import os 
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

def income_g(Tm = 30, d = 29, t = 30, group = 'treatment', incentive = 50, income= 0):
    if t != Tm:
        income = 0
    else:
        if group == 'control':
            income = 1000
        if group =='treatment':
            income = 500 + (incentive*max(0,d-10))
    return income


def Utility(L = 1,Tm = 30, d = 29, t = 30, beta = 0.03, mu = 4, P = 3, eps = 0):
    u = beta*income_g(Tm,d,t) + (mu + eps - P)*L
    return u

def VF(L = 1, 
       t = 30,
       d = 29,
       probfire = 0.0, 
       F = 0,
       eve_l = 0,
       eve_w = 0, 
       beta = 0.03,
       mu = 4):
    
        fired  = probfire*F
        vf_l = (Utility(L=1, d = d, t = t, beta = beta, mu = mu) + eve_l)
        vf_w = (Utility(L=0, d = d, t = t, beta = beta, mu = mu) + eve_w)
        worker = (1 - probfire)*(vf_l*L+ vf_w*(1-L))
        return max(fired,worker)
    

def exp_epsilon(eps_mean = 0 , eps_sd = 1, eps_thres = 0.5):
    e_exp = eps_mean + (eps_sd*norm.pdf((eps_thres-eps_mean)/eps_sd)/(1-norm.cdf((eps_thres-eps_mean)/eps_sd)));
    return e_exp

def model_solution(Tm = 30, beta = 0.03, mu = 4, P =3):
    # Probability matrix
    probs_w = np.empty((Tm,Tm)) 
    probs_w[:] = np.nan         
    
    probs_l = np.empty((Tm,Tm)) 
    probs_l[:] = np.nan         
    
    # Threshold matrix
    eps_t = np.empty((Tm,Tm)) 
    eps_t[:] = np.nan         
    
    # Expected values matrix for L = 1 
    eve_w  = np.empty((Tm,Tm)) 
    eve_w[:] = np.nan           
    
    # Expected values matrix
    eve_l  = np.empty((Tm,Tm)) 
    eve_l[:] = np.nan           
    
    # Value function matrix
    vf = np.empty((Tm,Tm)) 
    vf[:] = np.nan  
        
    #  SOLVING THE MODEL
    for t in range(Tm,-1,-1):
        print()
        for d in range(0,Tm):
            if d >= t:
                pass
            else:            
                if t == 30:
                    eps_t[t-1,d] = beta*(income_g(Tm=Tm, d = d+1, t = t) - income_g(Tm=Tm, d = d, t = t))- (mu-P)
                else:
                    eps_t[t-1,d] = vf[t,d+1] - vf[t,d] - (mu-P)
                probs_l[t-1,d] = 1 - norm.cdf(eps_t[t-1,d])
                probs_w[t-1,d] = 1 - probs_l[t-1,d]
                
                if t == 30:
                    eve_l[t-1,d] = 0 + exp_epsilon(eps_thres = eps_t[t-1,d])
                    eve_w[t-1,d] = 0
                else:
                    eve_l[t-1,d] = vf[t,d] + exp_epsilon(eps_thres = eps_t[t-1,d])
                    eve_w[t-1,d] = vf[t,d+1]
                
                vf[t-1,d] = probs_l[t-1,d]*(VF(L = 1, t = t, d = d, beta = beta, mu = mu, eve_l = eve_l[t-1,d])) + \
                probs_w[t-1,d]*(VF(L = 0, t = t, d = d+1, beta = beta, mu = mu, eve_w = eve_w[t-1,d])) 
    return probs_w, probs_l

'''
CODE TO EVALUATE OPTMIZERS
methods = [ 'Nelder-Mead','Powell', 'CG', 'BFGS', 'L-BFGS-B', \
            'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', \
            'trust-exact', 'trust-krylov']
dic = []
for m in methods:
    print(m)
    dics = {}
    start = time.perf_counter()
    resu = minimize(opt_likelihood, theta, method = M )
    end = time.perf_counter()
    dics['method'] = m
    dics['result'] = resu.x
    dics['time'] = end-start
    dic.append(dics)
    print("Elapsed (with compilation) = {}s".format((end - start)))

methods_frame = pd.DataFrame(dic)
'''

#%%[markdown]
#  Downloading the data and printing
# %%
sim_nb = 100
os.chdir('/Users/angelosantos/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/H_20223_ECON_7395_20093 - angelo_santos/DHR/data_created')
data = pd.read_pickle('simulation_'+str(sim_nb)+'_treatment.pkl')
data.head(100)
#%%[markdown]
# Transforming into numpy array
#%%
data = data.to_dict('records')
# %% 
def opt_likelihood(theta):
    probs_w, probs_l  = model_solution(beta = theta[0], mu = theta[1]);
    logL = 0
    for i in range(len(data)):
        t = int(data[i]['t'])
        d = int(data[i]['d'])
        L = int(data[i]['L'])
        likelihood = np.log(probs_w[t-1,d])*(1-L)+ np.log(probs_l[t-1,d])*L
        logL = logL + likelihood 
    return -logL


#%%

theta=[0.01,4.5]
opt_likelihood(theta)

start = time.perf_counter()
resu = minimize(opt_likelihood, theta, method = "SLSQP" )
end = time.perf_counter()
print('Optimun beta:' + str(round(resu.x[0],2)) + "\n" + 'Optimun mu:' + str(round(resu.x[1],2)))
print("Elapsed (with compilation) = {}s".format((end - start)))


# %%
