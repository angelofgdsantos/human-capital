#%%
'''

'''
import os
import time
from typing import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from dhr_model_solution import income_g, model_solution

# PLOTTING SETTINGS
plt.rcParams['figure.dpi'] = 500
plt.rcParams['savefig.dpi'] = 500
sns.set_style('ticks')
sns.despine(left=False, bottom=True)
sns.set_context("paper")

'''
Good colors for contour:
    - cool
    - viridis
    - spring
    - winter
'''

#%%[markdown]

## Functions
# Here we are going to define our functions for the Optimal policy solution 


#%%

def prob_ds(Tm = 30, probs_w = np.nan):
    
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

############################################################################### DEFINING MATRIX
def simulate_data(theta, M, I, gs = 1):
    np.random.seed(1996)
    d_means = np.empty((1,2))
    d_means[:] = np.nan         
    if gs == 1:
        group = ['treatment', 'control']
    else:
        group = ['treatment']
    for gnb, g in enumerate(group):
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

def exp_cost_days(Tm = 30, beta = 0.03, mu = 4, M = 10, I = 50, g = 'treatment'):
    frame = []
    probs_w = model_solution(Tm=Tm, g = g, I = I, M = M, beta = beta, mu = mu)[0]
    probs = prob_ds(probs_w = probs_w)
    incomes = np.empty((1,Tm))
    incomes[:] = np.nan
    ec = 0
    dc = 0
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
    Cp = np. empty((n2,n1))
    Dp = np. empty((n2,n1))

    for inb,incentive in enumerate(i):
        Ip[inb,:] = incentive
    for mn,money in enumerate(m):
        print(money)
        Mp[:,mn] = 30 - money


    Counter = 0
    for inb,incentive in enumerate(i):
        for mn,money in enumerate(m):
            print('Combination '+ str(counter) + ' of ' + str(n1*n2))
            fr = exp_cost_days(M=money, I=incentive)
            exc = fr.loc[fr['Day'] =='Expected Cost']['C'].item()
            exd = simulate_data(theta = [0.03,4], M=money, I=incentive)[0,0]
            Cp[inb,mn] = exc
            Dp[inb,mn] = exd
            counter += 1
    return Mp, Ip, Cp, Dp



# PLOT
def contours_plots(x,y,z,color = cm.cool, ap = 0.5, var = 'cost'):
    fig,ax=plt.subplots(1,1)
    contours = ax.contourf(x, y, z, 10, cmap = color, alpha = ap)
    ax.set_title('Expected Cost Contour')
    ax.set_xlabel('M = Days to be on money')
    ax.set_ylabel('I = Financial incentive')
    fig.colorbar(contours)
    os.chdir('/Users/angelosantos/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/H_20223_ECON_7395_20093 - angelo_santos/DHR/plots/')
    plotname = 'expected_'+var+'_shades.png'
    plt.savefig('/Users/angelosantos/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/H_20223_ECON_7395_20093 - angelo_santos/DHR/plots/'+plotname)
    plt.close()

    fig,ax=plt.subplots(1,1)
    contours = ax.contour(x, y, z, 10, cmap = color, alpha = 0.5)
    ax.clabel(contours, inline=True, fontsize = 7)
    ax.set_title('Expected '+var+ ' Contour')
    ax.set_xlabel('30 - M = Days out of money')
    ax.set_ylabel('I = Financial incentive')
    os.chdir('/Users/angelosantos/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/H_20223_ECON_7395_20093 - angelo_santos/DHR/plots/')
    plotname = 'expected_'+var+'_curves.png'
    plt.savefig('/Users/angelosantos/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/H_20223_ECON_7395_20093 - angelo_santos/DHR/plots/'+plotname)
    plt.close()

#%% 
## PARAMETERS

#%%
start          = time.perf_counter()
groups         = ['control','treatment']
mu_gs          = {'control' : 4, 'treatment' : 4 }
mu             = mu_gs['treatment']
P_gs           = {'control' : 3, 'treatment' : 3 }
P              = P_gs['treatment']
beta           = 0.03
eps_m          = 0 
eps_sd         = 1
Tm             = 30
I              = 50
M              = 10
g              = 'treatment'
np.random.seed(1996)

n1 = 15
n2 = 20
m = np.linspace(1,20,num = n1)
i = np.linspace(20,100, num = n2)
plots = []
for money in m:
    for incentive in i:
        dic = {}
        fr = exp_cost_days(M=money, I=incentive)
        exc = fr.loc[fr['Day'] =='Expected Cost']['C'].item()
        dic['I = Pay per day'] = incentive
        dic['M = In money days'] = money
        dic['Expected Cost'] = exc 
        dic['Average days'] = simulate_data(theta = [0.03,4], M=money, I=incentive)[0,0]
        plots.append(dic)

plots = pd.DataFrame(plots)
sns.lineplot(data = plots, x = plots.columns[1], y = plots.columns[2], hue = plots.columns[0], palette= 'nipy_spectral_r' ).set(title = 'Expected cost')    
sns.despine(left=False, bottom=False)
plotname = 'Exp_cost.png'
plt.savefig('/Users/angelosantos/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/H_20223_ECON_7395_20093 - angelo_santos/DHR/plots/'+plotname)
plt.close()

plots = pd.DataFrame(plots)
sns.lineplot(data = plots, x = plots.columns[1], y = plots.columns[3], hue = plots.columns[0], palette= 'nipy_spectral_r' ).set(title = 'Expected days in school')    
sns.despine(left=False, bottom=False)
plotname = 'Exp_days.png'
plt.savefig('/Users/angelosantos/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/H_20223_ECON_7395_20093 - angelo_santos/DHR/plots/'+plotname)
plt.close()


Mp, Ip, Cp, Dp = countour_data(n1 = 20, n2 = 20)
colors = {
    'cost' : cm.cool,
    'days' : cm.viridis
}

for var in ['cost','days']:
    if var == 'cost':
        v = Cp
    else:
        v = Dp
    contours_plots(x = Mp, y = Ip, z = v, var=var, color=colors[var])





# %%
