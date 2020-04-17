# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:40:58 2020

@author: Jinghong Chen
"""
from mcmc import MH_Sampler
import numpy as np
import timeit
from numpy.matlib import repmat
import matplotlib.pyplot as plt

    
def random_walk(x):
    mm = x.shape[0]
    dx = 0.5*np.random.randn(mm) 
    return x+dx

if __name__ == '__main__':
    cost_mat = np.loadtxt("data/london_n/cost_mat.txt")
    orig = np.loadtxt("data/london_n/P.txt")
    xd = np.loadtxt("data/london_n/xd0.txt")
    nn, mm = np.shape(cost_mat)
    theta = np.array([2., 0.5*0.7e7, .3/mm, 100., 1.3])
    
    calc = Calc(theta, orig, cost_mat)

    sample_length = 30000
    
    mcmc = MH_Sampler(h=None, T=random_walk, N=sample_length, x0=xd, calc=calc)
    
    tic = timeit.default_timer()
    mcmc.run()
    toc = timeit.default_timer()
    my_xx = mcmc.X[-2000:]
    my_mean = np.mean(my_xx, axis=0)
    my_var = np.var(my_xx, axis=0)
    my_cov = np.cov(my_xx,rowvar=False)
#    print('mean: ',mean)
#    print('var: ',var)
#    print('cov:',cov)
    print('xd: ',xd )
    print('my_mean:',my_mean)
    print(f"Sampling chain of {sample_length} takes {toc-tic} seconds")
    print('potential at my_mean:',calc.potential(my_mean))
    print('potential at xd:', calc.potential(xd))
    pot = np.zeros(2000)
    for i, x in enumerate(my_xx):
        pot[i] = calc.potential(x)
    print('mean of potential:',np.mean(pot))
    print('variance of potential:',np.var(pot))
    print('maximum potential:',np.max(pot))
    print('minimum potential:',np.min(pot))    
    
    

