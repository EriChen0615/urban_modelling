# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:43:49 2020

@author: asus
"""

import numpy as np
from playground import *

if __name__ == '__main__':
    """
    Examine if the hmc sample follows the target distribution/ if my hmc sample follows the target distribution
    """
    cost_mat = np.loadtxt("data/london_n/cost_mat.txt")
    orig = np.loadtxt("data/london_n/P.txt")
    xd = np.loadtxt("data/london_n/xd0.txt")
    nn, mm = np.shape(cost_mat)
    theta = np.array([2., 0.5*0.7e7, .3/mm, 100., 1.3])
    
    calc = Calc(theta, orig, cost_mat)
    sample = np.loadtxt('./output/hmc_samples0.5.txt')
    xx = sample[8000:]
    mean = np.mean(xx, axis=0)
    var = np.var(xx, axis=0)
    cov = np.cov(xx,rowvar=False)
#    print('mean: ',mean)
#    print('var: ',var)
#    print('cov:',cov)
#    


    my_sample = np.loadtxt('./my_output/py_hmc_samples_trial1_2.0.txt')
    my_xx = sample[5000:8000]
    my_mean = np.mean(my_xx, axis=0)
    my_var = np.var(my_xx, axis=0)
    my_cov = np.cov(my_xx,rowvar=False)
#    print('mean: ',mean)
#    print('var: ',var)
#    print('cov:',cov)
    
    print('mean:',mean)
    print('my_mean:',my_mean)
    print('potential at mean:', calc.potential(mean))
    print('potential at my_mean:',calc.potential(my_mean))
#    print('gradient at mean:', calc.grad(mean))
    print('potential at xd0:', calc.potential(xd))
#    print('gradient at xd0:', calc.grad(xd))
    pot = np.zeros(2000)
    for i, x in enumerate(sample[8000:]):
        pot[i] = calc.potential(x)
    print('potential mean:',np.mean(pot))
    print('potential var:',np.var(pot))
    print('potential max:',np.max(pot))
    print('potential min:',np.min(pot))    
