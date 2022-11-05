import numpy as np
import pandas as pd
from numba import njit

@njit
def ϵ(age = 16):
    ϵ_vec = [1,1,1,1,1]
    return ϵ_vec

def R_m(age = 16, edu = 12,
        X   = np.array([1,1,1,0,0]),
        end = np.array([8.8043,9.85,9.5,43948,6887]),
        e1  = np.array([0.0938,0.0189,0.0443,0,0]), 
        e2  = np.array([0.117,0.0674,0.3391,0,0]), 
        e3  = np.array([0.0077,0.1424,-2.99,0,0]),
        tc1 = 2983, tc2 = 26357,
        work = np.array([1,1,1,0,0])):
    
    ed = np.array([0,0,0,edu,0])
    ed12 = (ed>=12).astype(int)
    ed16 = (ed>=16).astype(int)
    
    R = work*np.exp(end + e1*edu + e2*X - e3*(X**2) + ϵ(age)) \
        + (1-work)*(end - tc1*ed12 - tc2*ed16 + ϵ(age))
    return R

@njit
def reward(D = [0,0,0,0,1]):  
    R = R_m()
    rew = sum(R*D)
    return rew

@njit
def VF(age, δ = 0.787, EV = np.nan):
    U = reward(R_m(age)) + δ*EV
    return U