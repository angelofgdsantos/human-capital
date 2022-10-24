import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from numba import jit, njit
from scipy.stats import norm
from matplotlib import pyplot as plt

def income_g(Tm = 30, d = 29, t = 30, g = 'treatment', I = 50, M = 10, income = 0):
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