#%%
############################################################################### PACKAGES
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.optimize import minimize

############################################################################### FUNCTIONS
def income_g(Tm = 30, d = 29, t = 30, g = 'treatment', I = 50, M = 10, income = 0):
    if t != Tm:
        income = 0
    else:
        if g == 'control':
            income = 1000
        if g =='treatment':
            income = 500 + (I*max(0,d-M))
    return income

def Utility(L = 1,Tm = 30, d = 29, t = 30, g = 'treatment', I = 50, M = 10, beta = 0.03, mu = 4, P = 3, eps = 0):
    u = beta*income_g(Tm = Tm, d = d, t = t, g = g, I = I, M = M) + (mu + eps - P)*L
    return u

def VF(L = 1, 
       t = 30,
       d = 29,
       g = 'treatment', 
       I = 50, 
       M = 10,
       probfire = 0.0, 
       F = 0,
       eve_l = 0,
       eve_w = 0, 
       beta = 0.03,
       mu = 4):
    
        fired  = probfire*F
        vf_l = (Utility(L=1, d = d, t = t, beta = beta, mu = mu, g=g, I = I, M = M) + eve_l)
        vf_w = (Utility(L=0, d = d, t = t, beta = beta, mu = mu, g=g, I = I, M = M) + eve_w)
        worker = (1 - probfire)*(vf_l*L+ vf_w*(1-L))
        return max(fired,worker)
    
def exp_epsilon(eps_mean = 0 , eps_sd = 1, eps_thres = 0.5):
    e_exp = eps_mean + (eps_sd*norm.pdf((eps_thres-eps_mean)/eps_sd)/(1-norm.cdf((eps_thres-eps_mean)/eps_sd)));
    return e_exp

def model_solution(Tm = 30, g= 'treatment', I = 50, M = 10, beta = 0.03, mu = 4, P = 3):
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
                    eps_t[t-1,d] = beta*(income_g(Tm=Tm, d = d+1, t = t,  g=g, I = I, M = M) - \
                                    income_g(Tm=Tm, d = d, t = t,  g=g, I = I, M = M))- (mu-P)
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
                
                vf[t-1,d] = probs_l[t-1,d]*(VF(L = 1, t = t, d = d, beta = beta, mu = mu,  g=g, I = I, M = M, eve_l = eve_l[t-1,d])) \
                          + probs_w[t-1,d]*(VF(L = 0, t = t, d = d+1, beta = beta, mu = mu,  g=g, I = I, M = M, eve_w = eve_w[t-1,d])) 
    return probs_w, probs_l, eps_t, eve_w, eve_l, vf

############################################################################### DEFINING MATRIX
def simulate_data(theta, M, I):
    np.random.seed(1996)
    d_means = np.empty((1,2))
    d_means[:] = np.nan         

    for gnb, g in enumerate(['treatment', 'control']):
        probs_w, probs_l, eps_t, eve_w, eve_l, vf = model_solution(beta = theta[0], mu = theta[1], g=g, M=M, I=I)


        ############################################################################### PARAMETERS - SIMULATION
        '''
        This simulation will create two tables:
            - Days worked in t (d)
            - Leisure decsion (L)
        '''
        sim_nb   = 100
        mu       = theta[1]
        P        = 3
        Tm       = 30
        eps_mean = 0 
        eps_sd   = 1

        ############################################################################### SIMULATION
        # CREATE MATRICES
        L_dec = np.empty((Tm,sim_nb))
        L_dec[:] = np.nan         

        d_dec = np.empty((Tm,sim_nb))
        d_dec[:] = np.nan         

        d1_dec = np.empty((Tm,sim_nb))
        d1_dec[:] = np.nan         

        e_dec = np.empty((Tm,sim_nb))
        e_dec[:] = np.nan         

        u_dec = np.empty((Tm,sim_nb))
        u_dec[:] = np.nan         

        # LOOP TO SIMULATION
        for s in range(sim_nb):
            d = 0
            for t in range(30):
                d_dec[t,s] = d
                if t != 29:
                    e_shock = np.random.normal(loc = eps_mean, scale = eps_sd)
                    w = vf[t+1,d+1]
                    l = e_shock + mu - P + vf[t+1,d]
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
                    e_shock = np.random.normal(loc = eps_mean, scale = eps_sd)
                    w = income_g(Tm,d+1,Tm,g=g, M=M, I=I)
                    l = e_shock + income_g(Tm,d,Tm,g=g, M=M, I=I)
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
                print('Simulation: ' + str(s) +'\n t = '+str(t)+'; L = '+str(dec))
                L_dec[t,s] = dec
                d1_dec[t,s] = d
                e_dec[t,s] = e_shock
                u_dec[t,s] = u
        d_mean = d1_dec[29,:].mean()
        d_means[0,gnb] = d_mean
    return d_means        

def root_calibration(theta):
    means = simulate_data(theta, M = 10, I = 50)
    print(means)
    f = ( means[0,1] - d_mean_c0)**2 + (means[0,0] - d_mean_t0)**2
    return f

#%%
############################################################################### PARAMETERS - MODEL
start = time.perf_counter()
groups         = ['control','treatment']
mu_gs          = {'control' : 4, 'treatment' : 4 }
mu             = mu_gs['treatment']
P_gs           = {'control' : 3, 'treatment' : 3 }
P              = P_gs['treatment']
beta           = 0.03
eps_m          = 0 
eps_sd         = 1
Tm             = 30
income         = np.nan
incentive      = 50
incentive_days = 10
np.random.seed(1996)

sim_nb = 100 
os.chdir('/Users/angelosantos/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/H_20223_ECON_7395_20093 - angelo_santos/DHR/data_created')
data_t = pd.read_pickle('simulation_'+str(sim_nb)+'_treatment.pkl')
data_c= pd.read_pickle('simulation_'+str(sim_nb)+'_control.pkl')
d_mean_t0 = data_t.loc[data_t['t'] == 30]['d+1'].mean()
d_mean_c0 = data_c.loc[data_c['t'] == 30]['d+1'].mean()
theta = [0.03,0]
root_calibration(theta)

#%%
start = time.perf_counter()
theta = [0.01,4]
resu = minimize(root_calibration,theta, method='Nelder-Mead')
end = time.perf_counter()
print('Optimun beta:' + str(round(resu.x[0],2)) + "\n" + 'Optimun mu:' + str(round(resu.x[1],2)))
print("Elapsed (with compilation) = {}s".format((end - start)))
#%%




# %%
