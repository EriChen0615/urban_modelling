# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 23:28:14 2020

@author: Jinghong Chen
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_autocorr(x):
    mean = np.mean(x)
    x_demean = x - mean    
    auto_corr = np.correlate(x_demean,x_demean,'same')
    auto_corr = auto_corr[auto_corr.size//2:]
    plt.bar(np.arange(len(auto_corr)), auto_corr/auto_corr[0])
    plt.title('Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.show()
    print(f'Integrated correlation is {0.5+np.sum(auto_corr/auto_corr[0])}')
