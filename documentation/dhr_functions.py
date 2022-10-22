
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from numba import jit, njit
from scipy.stats import norm
from matplotlib import pyplot as plt

def income_g(Tm = 30, d = 29, t = 30, g = 'treatment', I = 50, M = 10, income = 0):
    """_summary_

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
