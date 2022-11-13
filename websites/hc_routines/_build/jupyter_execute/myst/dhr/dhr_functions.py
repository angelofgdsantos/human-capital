#!/usr/bin/env python
# coding: utf-8

# # Functions
# 
# These are the packages that you will need to run the functions defined here. We will call this file using `dhr.func` sintax in the following routines.

# In[1]:


import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from numba import jit, njit
from scipy.stats import norm
from matplotlib import pyplot as plt


# numba is a compiler developed for python and can make your codes faster. See more about it [here](https://numba.pydata.org).

# In[2]:


def income_g(t = 30, d = 29, g = 'treatment', I = 50, M = 10, income = 0, Tm = 30):
    """

    This functions will calculte the income for a particular teacher in the model 

    Parameters
    ----------
    Tm : int, optional
        Month total days, by default 30
    d : int, optional
        Days worked before de current day (t), by default 29
    t : int, optional
        Current day, by default 30
    g : str, optional
        Group in wich the teacher is, can be 'control' or 'treatment', by default 'treatment'
    I : int, optional
        The amount of incetive per addional day on the money, by default 50
    M : int, optional
        Days to be on the money, by default 10
    income : int, optional
        Income in the current period (t), by default 0

    Returns
    -------
    float
        This is the income of the teacher in the last day of the month.

    """
    if t != Tm:
        income = 0
    else:
        if g == 'control':
            income = 1000
        if g =='treatment':
            income = 500 + (I*max(0,d-M))
    return income

def Utility(L = 1,Tm = 30, d = 29, t = 30, g = 'treatment', I = 50, M = 10, beta = 0.03, mu = 4, P = 3, eps = 0):
    """
    
    Utility function

    Parameters
    ----------
    L : int, optional
        Leisure decision, by default 1
    Tm : int, optional
        Month total days, by default 30
    d : int, optional
        Days worked before de current day (t), by default 29
    t : int, optional
        Current day, by default 30
    g : str, optional
        Group in wich the teacher is, can be 'control' or 'treatment', by default 'treatment'
    I : int, optional
        The amount of incetive per addional day on the money, by default 50
    M : int, optional
        Days to be on the money, by default 10
    beta : float, optional
        Coefficient to translate income into utility, by default 0.03
    mu : int, optional
        Shifter of leisure value, by default 4
    P : int, optional
        Non-pecuniary cost, by default 3
    eps : int, optional
        Schock to leisure, by default 0

    Returns
    -------
    float
        Utility from leisure decision

    """
    u = beta*income_g(Tm = Tm, d = d, t = t, g = g, I = I, M = M) + (mu + eps - P)*L
    return u

def VF(L = 1, t = 30, d = 29, g = 'treatment', I = 50, M = 10, probfire = 0.0, F = 0, eve_l = 0, eve_w = 0, beta = 0.03, mu = 4):
    """

    Value function 
    
    Parameters
    ----------
    L : int, optional
        Leisure decision, by default 1
    t : int, optional
        Current day, by default 30
    d : int, optional
        Days worked before de current day (t), by default 29
    g : str, optional
        Group in wich the teacher is, can be 'control' or 'treatment', by default 'treatment'
    I : int, optional
        The amount of incetive per addional day on the money, by default 50
    M : int, optional
        Days to be on the money, by default 10
    probfire : float, optional
        Probability of getting fired, by default 0.0
    F : int, optional
        _description_, by default 0
    eve_l : int, optional
        Expected value of leisure, by default 0
    eve_w : int, optional
        Expected value of working, by default 0
    beta : float, optional
        Coefficient to translate income into utility, by default 0.03
    mu : int, optional
        Shifter of leisure value, by default 4
    
    Returns
    -------
    float
        Maximum value between working and leisure

    """
    fired  = probfire*F
    vf_l = (Utility(L=1, d = d, t = t, beta = beta, mu = mu, g=g, I = I, M = M) + eve_l)
    vf_w = (Utility(L=0, d = d, t = t, beta = beta, mu = mu, g=g, I = I, M = M) + eve_w)
    worker = (1 - probfire)*(vf_l*L+ vf_w*(1-L))
    return max(fired,worker)
    
def exp_epsilon(eps_mean = 0 , eps_sd = 1, eps_thres = 0.5):
    """

    Truncated value

    Parameters
    ----------
    eps_mean : int, optional
        mean, by default 0
    eps_sd : int, optional
        standart deviation, by default 1
    eps_thres : float, optional
        Threshold calculated by the proba, by default 0.5

    Returns
    -------
    float
        Expected value for the truncated distribution

    """
    e_exp = eps_mean + (eps_sd*norm.pdf((eps_thres-eps_mean)/eps_sd)/(1-norm.cdf((eps_thres-eps_mean)/eps_sd)));
    return e_exp

def model_solution(Tm = 30, g= 'treatment', I = 50, M = 10, beta = 0.03, mu = 4, P = 3):
    """

    Solves the model

    Parameters
    ----------
    Tm : int, optional
        Month total days, by default 30
    g : str, optional
        Group in wich the teacher is, can be 'control' or 'treatment', by default 'treatment'
    I : int, optional
        The amount of incetive per addional day on the money, by default 50
    M : int, optional
        Days to be on the money, by default 10
    beta : float, optional
        Coefficient to translate income into utility, by default 0.03
    mu : int, optional
        Shifter of leisure value, by default 4
    P : int, optional
        Non-pecuniary cost, by default 3
    
    Returns
    -------
    arrays
        probs_w : Matrix of probabilities of working

        probs_l : Matrix of probabilities of leisure

        eps_t : Matrix of thresholds

        eve_w : Matrix of expected value of working 
        
        eve_l : Matrix of expected value of leisure 
        
        vf : Value function matrix

    """
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
                    eps_t[t-1,d] = beta*(income_g(Tm=Tm, d = d+1, t = t,  g=g, I = I, M = M) -                                     income_g(Tm=Tm, d = d, t = t,  g=g, I = I, M = M))- (mu-P)
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
                
                vf[t-1,d] = probs_l[t-1,d]*(VF(L = 1, t = t, d = d, beta = beta, mu = mu,  g=g, I = I, M = M, eve_l = eve_l[t-1,d]))                           + probs_w[t-1,d]*(VF(L = 0, t = t, d = d+1, beta = beta, mu = mu,  g=g, I = I, M = M, eve_w = eve_w[t-1,d])) 
    return probs_w, probs_l, eps_t, eve_w, eve_l, vf

def simulate_data(theta = [0.03,4], M = 10, I = 50, gs = 0, P = 3, Tm = 30, sim_nb = 100, eps_mean = 0, eps_sd = 1):
    """
    
    This functions simulate individuals using the model solution

    Args:
        theta (list, optional): This is a list with two elements, beta and mu, respectively. Defaults to [0.03,4].
        M (int, optional): Days to be on money. Defaults to 10.
        I (int, optional): Financial Incentive. Defaults to 50.
        gs (int, optional): Dummy to simulate control and treatment. Defaults to 0.
        P (int, optional): Non pecuniary cost to leisure. Defaults to 3.
        Tm (int, optional): Total days in the month. Defaults to 30.
        sim_nb (int, optional): Number of simulated individuals. Defaults to 100.
        eps_mean (int, optional): Shock mean. Defaults to 0.
        eps_sd (int, optional): Shock Standart error. Defaults to 1.

    Returns:
        Tuple: Returns a tuple with:
           - d_means: Means from datset, treatment and control. This is used for estimation.
           - L_dec: Leisure decisions
           - d_dec: Days worked in t
           - d1_dec: Days worked in t+1
           - e_dec: Realized shocks faced
           - u_dec: Utility matrix due decisions
           - groups_data: Dictionary for control and treatment simulation results
    """
    np.random.seed(1996)
    d_means = np.empty((1,2))
    d_means[:] = np.nan   
    groups_data = {}
    if gs == 1:
        group = ['treatment', 'control']
        groups_data['treatment'] = {}     
        groups_data['control'] = {}     
    else:
        group = ['treatment']
        groups_data['treatment'] = {}     
    for gnb, g in enumerate(group):
        _, _, _, _, _, vf = model_solution(beta = theta[0], mu = theta[1], g=g, M=M, I=I)


        ############################################################################### PARAMETERS - SIMULATION
        '''
        This simulation will create two tables:
            - Days worked in t (d)
            - Leisure decsion (L)
        '''
        mu = theta[1]
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
                    e_shock = np.random.normal(eps_mean, eps_sd)
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
                    e_shock = np.random.normal(eps_mean, eps_sd)
                    w = income_g(t+1, d+1, g=g, M=M, I=I)
                    l = e_shock + income_g(t+1, d, g=g, M=M, I=I)
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
                L_dec[t,s] = dec
                d1_dec[t,s] = d
                e_dec[t,s] = e_shock
                u_dec[t,s] = u
        d_mean = d1_dec[29,:].mean()
        d_means[0,gnb] = d_mean
        groups_data[g]['L_dec'] = L_dec
        groups_data[g]['d_dec'] = d_dec
        groups_data[g]['d1_dec'] = d1_dec
        groups_data[g]['e_dec'] = e_dec
        groups_data[g]['u_dec'] = u_dec
    return d_means, L_dec, d_dec, d1_dec, e_dec, u_dec, groups_data

def dataframe_simulation(theta = [0.03,4], M = 10, I = 50, sim_nb = 100):
    """
    
    This function put the simulation in datasets

    Args:
        sim_nb (int, optional): Number of individuals. Defaults to 100.

    Returns:
        Dict: Dictionary with dataframe simulated for treatment and control
        
    """
    datasets = {}
    _, _, _, _, _, _, datas = simulate_data(theta, M, I,  gs=1, sim_nb=sim_nb)
    for g in ['treatment','control']:
        data = datas[g]
        simulated_data = []
        for i in range(sim_nb):
            s = {}
            s['id'] = i
            s['t']  = list(range(1,31))
            s['d+1']  = list(data['d1_dec'][:,i])
            s['d']    = list(data['d_dec'][:,i])
            s['L']  = list(data['L_dec'][:,i])
            s['eps']  = list(data['e_dec'][:,i])
            s['U']  = list(data['u_dec'][:,i])
            s = pd.DataFrame(s)
            simulated_data.append(s)    
        simulated_data = pd.concat(simulated_data, axis = 0).reset_index().drop('index', axis = 1)
        datasets[g] = simulated_data
    return datasets

def discrete_probs(theta = [0.03,4], Tm = 30, M=10, I = 50):
    """
    
    This function creates a matrix with the probabilities 

    Args:
        theta (list, optional): This is a list with two elements, beta and mu, respectively. Defaults to [0.03,4].
        Tm (int, optional): Total days in the month. Defaults to 30.

    Returns:
        DataFrame: Matrix with probabilities
    """
    probs_w, _, _, _, _, _ = model_solution(beta = theta[0], mu = theta[1], M = M, I = I)
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
    return prob_sim

def Log_likelihood(data, b_guess, mu_guess, n = 10, group = 'treatment'):
    """
    
    This function does the log likehood iteration for the estimation using the parameters grid

    Args:
        data (_type_): Data simulated for the group of choice
        b_guess (_type_): Lower bound for beta guess
        mu_guess (_type_): Lower bound for mu guess
        n (int, optional): Size of the grid. Defaults to 10.
        group (str, optional): Group of simulation. Defaults to 'treatment'.

    Returns:
        Dataframe: Dataframe with parameters and log likelihood result
    """
    log_values = []
    sim = 1
    for b in b_guess:
        for m in mu_guess:
            dic = {}
            probs_w, probs_l, _, _, _, _, = model_solution(beta = b, mu = m, g = 'treatment');
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

def opt_likelihood(data,theta):
    """
    
    This is the Log likelihood used for the optimizer

    Args:
        data (_type_): Dataset used
        theta (_type_): Initial guess

    Returns:
        float: Log likelihood result
    """
    probs_w, probs_l  = model_solution(beta = theta[0], mu = theta[1]);
    logL = 0
    for i in range(len(data)):
        t = int(data[i]['t'])
        d = int(data[i]['d'])
        L = int(data[i]['L'])
        likelihood = np.log(probs_w[t-1,d])*(1-L)+ np.log(probs_l[t-1,d])*L
        logL = logL + likelihood 
    return -logL

def root_calibration(theta):
    d_mean_c0 = 5.2
    d_mean_t0 = 20.82
    """
    
    This function find the parameters that calibrate the model

    Args:
        theta (_type_): Initial values for the parameters

    Returns:
        List: Parameters combination in a list
    """
    means, _, _, _, _, _, _, = simulate_data(theta, M = 10, I = 50, gs = 1)
    print(means)
    f = ( means[0,1] - d_mean_c0)**2 + (means[0,0] - d_mean_t0)**2
    return f

def exp_cost_days(Tm = 30, beta = 0.03, mu = 4, M = 10, I = 50, g = 'treatment'):
    """
    
    This function creates a DataFame with the expected cost conditional on the expected days worked by teachers

    Args:
        Tm (int, optional): Total number of days in the month. Defaults to 30.
        beta (float, optional): Parameter value of beta. Defaults to 0.03.
        mu (int, optional): Parameter value of mu. Defaults to 4.
        M (int, optional): Days to be on money. Defaults to 10.
        I (int, optional): Financial incentive for additional day worked. Defaults to 50.
        g (str, optional): Group simulated. Defaults to 'treatment'.

    Returns:
        DataFrame: Returns a data frame with costs conditional on expected days worked 
    """
    frame = []
    probs_w = model_solution(Tm=Tm, g = g, I = I, M = M, beta = beta, mu = mu)[0]
    probs = discrete_probs(theta=[beta, mu], M = M, I = I)
    incomes = np.empty((1,Tm))
    incomes[:] = np.nan
    ec = 0
    for day in range(30):
        dic ={}
        pw = (income_g(M = M, I = I, g = g, d = day+1)) * (probs[29,day] *probs_w[29,day])
        pl = (income_g(M = M, I = I, g = g, d = day))* (probs[29,day]*(1 - probs_w[29,day]))
        c = pw + pl
        dic['Day'] = day
        dic['C'] = c
        frame.append(dic)
        ec = ec + c
    dic={}
    dic['Day'] = 'Expected Cost'
    dic['C'] = ec
    frame.append(dic)
    frame = pd.DataFrame(frame)
    return frame

def countour_data(n1,n2):    
    m = np.linspace(1,20,num = n1)
    i = np.linspace(20,100, num = n2)

    counter = 0 
    Ip = np.empty((n2,n1))
    Mp = np.empty((n2,n1))
    Cp = np.empty((n2,n1))
    Dp = np.empty((n2,n1))

    for inb,incentive in enumerate(i):
        Ip[inb,:] = incentive
    for mn,money in enumerate(m):
        print(money)
        Mp[:,mn] = 30 - money


    for inb,incentive in enumerate(i):
        for mn,money in enumerate(m):
            print('Combination '+ str(counter) + ' of ' + str(n1*n2))
            fr = exp_cost_days(M=money, I=incentive)
            exc = fr.loc[fr['Day'] =='Expected Cost']['C'].item()
            exd = simulate_data(theta = [0.03,4], M=money, I=incentive)[0][0,0]
            Cp[inb,mn] = exc
            Dp[inb,mn] = exd
            counter += 1
    return Mp, Ip, Cp, Dp

# PLOTTING
def simulation_plot(s_plot = 5, nb_sim = 100, p = 'nipy_spectral_r', sim_nb = [0,20,40,60,80,99], group = 'treatment'):
    d_means, L_dec, d_dec, d1_dec, e_dec, u_dec, _= simulate_data()
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
    plt.savefig('/Users/angelosantos/Documents/GitHub/human-capital/images/dhr/'+plotname)
    plt.close()

def t_graphs(t_plot = 5, p = 'nipy_spectral_r', ts = [0,10,15,20,25,29], group = 'treatment'):
    probs_w, probs_l, eps_t, eve_w, eve_l, vf = model_solution()
    prob_sim = discrete_probs()
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
    plt.savefig('/Users/angelosantos/Documents/GitHub/human-capital/images/dhr/'+plotname)
    plt.close()
    
    # Plot Probability of days worked
    sns.lineplot(data = ts_frames, x = ts_frames.columns[0], y = ts_frames.columns[2], hue = ts_frames.columns[-1], palette= 'nipy_spectral_r' ).set(title = 'Prob(d|t), x=days-attended, y=probability')    
    sns.despine(left=False, bottom=False)
    if group == 'treatment':
        plotname = 'prob_days_worked.png'
    else:
        plotname = 'prob_days_worked_control.png'
    plt.savefig('/Users/angelosantos/Documents/GitHub/human-capital/images/dhr/'+plotname)
    plt.close()

    # Plot Probability of working
    sns.lineplot(data = ts_frames, x = ts_frames.columns[0], y = ts_frames.columns[3], hue = ts_frames.columns[-1], palette= 'nipy_spectral_r' ).set(title = 'Prob(L=0|t,d), x=days-attended, y=probability')    
    sns.despine(left=False, bottom=False)
    if group == 'treatment':
        plotname = 'prob_working.png'
    else:
        plotname = 'prob_working_control.png'
    plt.savefig('/Users/angelosantos/Documents/GitHub/human-capital/images/dhr/'+plotname)
    plt.close()
        
def contours_plots(x,y,z,color = cm.cool, ap = 0.5, var = 'cost'):
    fig,ax=plt.subplots(1,1)
    contours = ax.contourf(x, y, z, 10, cmap = color, alpha = ap)
    ax.set_title('Expected Cost Contour')
    ax.set_xlabel('M = Days to be on money')
    ax.set_ylabel('I = Financial incentive')
    fig.colorbar(contours)
    plotname = 'expected_'+var+'_shades.png'
    plt.savefig('/Users/angelosantos/Documents/GitHub/human-capital/images/dhr'+plotname)
    plt.close()

    fig,ax=plt.subplots(1,1)
    contours = ax.contour(x, y, z, 10, cmap = color, alpha = 0.5)
    ax.clabel(contours, inline=True, fontsize = 7)
    ax.set_title('Expected '+var+ ' Contour')
    ax.set_xlabel('30 - M = Days out of money')
    ax.set_ylabel('I = Financial incentive')
    plotname = 'expected_'+var+'_curves.png'
    plt.savefig('/Users/angelosantos/Documents/GitHub/human-capital/images/dhr'+plotname)
    plt.close()

