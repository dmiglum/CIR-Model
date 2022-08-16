
# CIR Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def CIR_process(r0, theta, kappa, sigma, T = 1., N = 100, seed = 0):
    if seed != 0:
        np.random.seed(seed)

    dt = T/N
    rates = [r0]
    for i in range(N):
        dr = kappa*(theta-rates[-1])*dt + sigma*np.sqrt(rates[-1])*np.random.normal(size = 1, scale = np.sqrt(dt))
        rates.append(rates[-1] + dr)
    
    return(pd.DataFrame(data = rates, index = [x*dt for x in range(N+1)] ))


def CIR_A(theta, kappa, sigma, T):
    gamma = np.sqrt(kappa**2+2*sigma**2)
    tmp = -2*kappa*theta/sigma**2
    tmp = tmp*np.log( 2*gamma*np.exp((gamma+kappa)*T/2) / ( (gamma+kappa)*(np.exp(gamma*T)-1) +2*gamma ))
    return(tmp)


def CIR_B(kappa, sigma, T):
    gamma = np.sqrt(kappa**2+2*sigma**2)
    tmp = 2*(np.exp(gamma*T)-1) /((gamma+kappa)*(np.exp(gamma*T)-1) +2*gamma )
    return(tmp)


def CIR_discount_curve(r0, theta, kappa, sigma, T = 10, N = 50):
    t = np.linspace(0,T,N)
    A = np.array([CIR_A(theta, kappa, sigma, T) for T in t])
    B = np.array([CIR_B(kappa, sigma, T) for T in t])
    tmp = np.exp(-A-B*r0)
    tmp = pd.DataFrame(data = tmp, index = t)
    return(tmp)


def CIR_yield_curve(r0, theta, kappa, sigma, T = 10, N = 50):
    discount_curve = CIR_discount_curve(r0, theta, kappa, sigma, T, N)
    t = discount_curve.index
    y = [r0]
    for x in t[1:]:
        y.append(-np.log(discount_curve.loc[x,:].values[0])/x)
    tmp = pd.DataFrame(data = y, index = t)
    return(tmp)


if __name__ == '__main__':

    r0, theta, kappa, sigma = [0.03, 0.1, 0.15, 0.12]
    
    T , N, mc = [10., 100, 5]
    
    plot = True
    
    rates = CIR_process(r0, theta, kappa, sigma, T, N)
    for i in range(mc-1):
        rates = pd.concat([rates, CIR_process(r0, theta, kappa, sigma, T, N)], axis = 1)

    DFC = CIR_discount_curve(r0, theta, kappa, sigma, T)    
    YC = CIR_yield_curve(r0, theta, kappa, sigma, T)
    
    if plot == True:
        
        fig, axes = plt.subplots(3,1, sharex = True)
        
        axes[0].plot(rates, alpha = 0.25)
        axes[0].axhline(theta, c = 'black', linestyle = ':')
        
        axes[1].plot(DFC)
        
        axes[2].plot(YC)
        
        [x.grid() for x in axes]
        fig.tight_layout()
    
    kappa = 0.15
    sigma = 0.12
    r0 = 3./100

    def test_function(theta):
        return((CIR_discount_curve(r0, theta, kappa, sigma, 1, N = 2).iloc[1,0]**-1-1 - 0.05)**2)

    res = minimize(test_function, 0.05)
    print(res.x)