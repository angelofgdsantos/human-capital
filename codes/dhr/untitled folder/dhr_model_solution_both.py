
"""
Created on Thu Sep 15 18:06:45 2022

@author: angelosantos


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
############################################################################### PACKAGES
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from matplotlib import pyplot as plt

############################################################################### PLOTTING SETTINGS
plt.rcParams['figure.dpi'] = 500
plt.rcParams['savefig.dpi'] = 500
sns.set_style('ticks')
sns.despine(left=False, bottom=True)
sns.set_context("paper")

'''
Simulation graphs palletes:
    - Dark2
    - gist_ncar
    - gist_ncar_r
    - gist_rainbow 
    - gist_rainbow_r *
    - gnuplot
    - icefire *
    - nipy_spectral_r *
'''


############################################################################### FUNCTIONS
def income_g(Tm = 30, d = 29, t = 30, group = 'treatment', incentive = 50, income= 0):
    if t != Tm:
        income = 0
    else:
        if group == 'control':
            income = 1000
        if group =='treatment':
            income = 500 + (incentive*max(0,d-10))
    return income

def Utility(L = 1,Tm = 30, d = 29, t = 30, beta = 0.03, mu = 4, P = 3, eps = 0, group = 'treatment'):
    u = beta*income_g(Tm,d,t, group) + (mu + eps - P)*L
    return u

def VF(L = 1, 
       t = 30,
       d = 29,
       probfire = 0.0, 
       F = 0,
       eve_l = 0,
       eve_w = 0, 
       beta = 0.03,
       mu = 4,
       group = 'treatment'
       ):
    
        fired  = probfire*F
        vf_l = (Utility(L=1, d = d, t = t, beta = beta, mu = mu, group = group) + eve_l)
        vf_w = (Utility(L=0, d = d, t = t, beta = beta, mu = mu, group = group) + eve_w)
        worker = (1 - probfire)*(vf_l*L+ vf_w*(1-L))
        return max(fired,worker)
    
def exp_epsilon(eps_mean = 0 , eps_sd = 1, eps_thres = 0.5):
    e_exp = eps_mean + (eps_sd*norm.pdf((eps_thres-eps_mean)/eps_sd)/(1-norm.cdf((eps_thres-eps_mean)/eps_sd)));
    return e_exp

def model_solution(Tm = 30, beta = 0.03, mu = 4, group = 'treatment'):
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
                    eps_t[t-1,d] = beta*(income_g(Tm=Tm, d = d+1, t = t, group = group) - income_g(Tm=Tm, d = d, t = t, group = group))- (mu-P)
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
                
                vf[t-1,d] = probs_l[t-1,d]*(VF(L = 1, t = t, d = d, beta = beta, mu = mu, eve_l = eve_l[t-1,d], group = group)) + \
                probs_w[t-1,d]*(VF(L = 0, t = t, d = d+1, beta = beta, mu = mu, eve_w = eve_w[t-1,d], group = group)) 
    return probs_w, probs_l, eps_t, eve_w, eve_l, vf

# PLOTTING
def simulation_plot(s_plot = 5, nb_sim = 100, p = 'nipy_spectral_r', sim_nb = [0,20,40,60,80,99], group = 'treatment'):
    sims = []
    for s in sim_nb:    
        plot_frame = {}
        plot_frame['Day of month (t)'] = list(range(1,31))
        plot_frame['Days worked until t (d)'] = list(d_dec[:,s])
        sim = pd.DataFrame(plot_frame)
        sim['Simulation'] = s
        sims.append(sim)
    
    sims = pd.concat(sims, axis = 0).reset_index().drop('index', axis = 1)
    sns.lineplot(data = sims, x = sims.columns[0], y = sims.columns[1], hue = sims.columns[2], palette= 'nipy_spectral_r' ).set(title = 'Simulation Histories')    
    sns.despine(left=False, bottom=False)
    if group == 'treatment':
        plotname = 'simulation_histories_'+str(s_plot)+'_sim_control.png'
    else:
        plotname = 'simulation_histories_'+str(s_plot)+'_sim_control.png'
    plt.savefig('/Users/angelosantos/Documents/GitHub/human-capital/codes/dhr/plots/'+plotname)
    plt.close()

def t_graphs(t_plot = 5, p = 'nipy_spectral_r', ts = [0,10,15,20,25,29], group = 'treatment'):
    t_frames = []
    for t in ts:
        plot_frame = {}
        plot_frame['Days worked so far'] = list(range(t+1))
        plot_frame['EV (t,d)'] = list(vf[t,:t+1])
        plot_frame['Probability of days worked contional on t'] = list(prob_sim[t,:t+1])
        plot_frame['Probability of working given (t,d)'] = list(probs_w[t,:t+1])
        sim = pd.DataFrame(plot_frame)
        sim['Last day of month'] = t
        t_frames.append(sim) 
    
    # PLot EVs
    ts_frames = pd.concat(t_frames, axis = 0).reset_index().drop('index', axis = 1)
    sns.lineplot(data = ts_frames, x = ts_frames.columns[0], y = ts_frames.columns[1], hue = ts_frames.columns[-1], palette= 'nipy_spectral_r' ).set(title = 'EV(t,d), x=days-attended, z=days-attended-so-far')    
    sns.despine(left=False, bottom=False)
    if group == 'treatment':
        plotname = 'EVs.png'
    else:
        plotname = 'EVs_control.png'
    plt.savefig('/Users/angelosantos/Documents/GitHub/human-capital/codes/dhr/plots/'+plotname)
    plt.close()
    
    # Plot Probability of days worked
    sns.lineplot(data = ts_frames, x = ts_frames.columns[0], y = ts_frames.columns[2], hue = ts_frames.columns[-1], palette= 'nipy_spectral_r' ).set(title = 'Prob(d|t), x=days-attended, y=probability')    
    sns.despine(left=False, bottom=False)
    if group == 'treatment':
        plotname = 'prob_days_worked.png'
    else:
        plotname = 'prob_days_worked_control.png'
    plt.savefig('/Users/angelosantos/Documents/GitHub/human-capital/codes/dhr/plots/'+plotname)
    plt.close()

    # Plot Probability of working
    sns.lineplot(data = ts_frames, x = ts_frames.columns[0], y = ts_frames.columns[3], hue = ts_frames.columns[-1], palette= 'nipy_spectral_r' ).set(title = 'Prob(L=0|t,d), x=days-attended, y=probability')    
    sns.despine(left=False, bottom=False)
    if group == 'treatment':
        plotname = 'prob_working.png'
    else:
        plotname = 'prob_working_control.png'
    plt.savefig('/Users/angelosantos/Documents/GitHub/human-capital/codes/dhr/plots/'+plotname)
    plt.close()

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


############################################################################### DEFINING MATRIX
for g in ['treatment', 'control']:
    probs_w, probs_l, eps_t, eve_w, eve_l, vf = model_solution(group=g)


    ############################################################################### PARAMETERS - SIMULATION
    '''
    This simulation will create two tables:
        - Days worked in t (d)
        - Leisure decsion (L)
    '''
    sim_nb   = 100
    mu       = 4
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
                w = income_g(Tm,d+1,Tm,group=g)
                l = e_shock + income_g(Tm,d,Tm,group=g)
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
    os.chdir('/Users/angelosantos/Documents/GitHub/human-capital/codes/dhr/data')
    simulated_data.to_pickle('simulation_'+str(sim_nb)+'_'+g+'.pkl')
    simulated_data.to_csv('simulation_'+str(sim_nb)+'_'+g+'.csv', index = False)



    # MEASURING THE PROBABILITIES
    '''
    In this part of the code we will compute the probabilities using the simulation
        - We have for cases for t+1:
            L = 0 when in t and d = d   --> leads to d+1 = d   in t+1
            L = 1 when in t and d = d   --> leads to d+1 = d+1 in t+1
            L = 1 when in t and d = d-1 --> leads to d+1 = d   in t+1 
            L = 0 when in t and d = d-1 --> leads to d = d-1   in t+1
        - PS:
            if t = 1 --> d must be equal to zero --> P(1,0) = 1
    '''

    prob_sim = np.zeros((Tm,Tm))

    prob_sim[0,0] = 1 #P(t = 1, d = 0)

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
            
    ############################################################################### PLOTTING 
    simulation_plot(group=g)

    # Expected values and probs    
    t_graphs(group=g)
        
    end = time.perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
