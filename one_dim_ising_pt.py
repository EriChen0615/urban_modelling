# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 23:31:43 2020

@author: asus
"""

from ising import Ising_1D_Calc
from mcmc import PT_Sampler
from analysis import plot_autocorr
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt


N = 100

def flip_one_bit(x):
    new_x = x.copy()
    flip_num = 2
    new_x[np.random.randint(N,size=flip_num)] *= -1
    return new_x

def plot_config(x):
    Xv = np.arange(x.size)
    Yv = np.ones(x.size)
    white_ind = x==1
    black_ind = x==-1
    plt.plot(Xv[white_ind],Yv[white_ind],'rs',linewidth=1, markersize=1)
    plt.plot(Xv[black_ind],Yv[black_ind],'bs',linewidth=1, markersize=1)
    plt.title('Configuration')
    plt.show()
    
        
if __name__ == '__main__':
    J = -1.0
    B = 0
    kB = 1.38064852e-23
    d = 4 # number of chains
    temp0 = abs(J/kB)
    temps = np.array([0.1*(i+1)*temp0 for i in range(d)])
    calc = Ising_1D_Calc(temp0, J, B)
    sample_length = 2000
#    x0 = np.random.choice([-1,1],size=(N))
    x0 = np.ones((N)) # block initialization (test with negative J)
#    x0 = repmat(np.array([[1,-1]),1,N//2) # scattered initialization (test with positive J)
    
#    mcmc = MH_Sampler(h=None, T=flip_one_bit, N=sample_length, x0=x0, calc=calc)
    mcmc_pt = PT_Sampler(h=None, T=flip_one_bit, N=sample_length, x0=x0, temps=temps, swap_p=0.7, calc=calc)
    
    energy_pt = np.empty((sample_length, d))
    magnet_pt = np.empty((sample_length, d))
    for n in range(sample_length):
        mcmc_pt.step()
        for i in range(d):
            energy_pt[n,i] = calc.potential(mcmc_pt.x[i])
            magnet_pt[n,i] = calc.avg_magnet(mcmc_pt.x[i], N)
        
    plt.plot(np.arange(sample_length), energy_pt[:,0])
    plt.title('PT Energy')
    plt.show()
    plt.plot(np.arange(sample_length), magnet_pt[:,0])
    plt.title('PT Average Magnetisation')
    plt.show()
#    mcmc_pt.run()
#    plot_config(mcmc_pt.X[-100,0])
#    plot_config(mcmc_pt.x[0])
    plot_autocorr(magnet_pt[-200:,0])
    
    mcmc_mh = MH_Sampler(h=None, T=flip_one_bit, N=sample_length, x0=x0, calc=calc)
    
    energy_mh = np.empty((sample_length))
    magnet_mh = np.empty((sample_length))
    for n in range(sample_length):
        mcmc_mh.step()
        energy_mh[n] = calc.potential(mcmc_mh.x)
        magnet_mh[n] = calc.avg_magnet(mcmc_mh.x, N)
        
    plt.plot(np.arange(sample_length), energy_mh)
    plt.title('MH Energy')
    plt.show()
    plt.plot(np.arange(sample_length), magnet_mh)
    plt.title('MH Average Magnetisation')
    plt.show()
#    mcmc_mh.run()
#    plot_config(mcmc_mh.X[-100:])
#    plot_config(mcmc_mh.x)
    plot_autocorr(magnet_mh[-200:])
    
    
    
    
    

    
    
    

