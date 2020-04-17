# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:40:58 2020

@author: Jinghong Chen
"""
from ising import Ising_2D_Calc
from mcmc import MH_Sampler
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt


N = 50

def random_flip(x, N):
    new_x = x.copy()
    new_x[np.random.randint(N, size=10), np.random.randint(N, size=10)] *= -1
    return new_x

def flip_one_bit(x):
    new_x = x.copy()
    flip_num = 5
    new_x[np.random.randint(N,size=flip_num), np.random.randint(N,size=flip_num)] *= -1
    return new_x

def plot_config(x):
    mm, nn = x.shape
    Xv, Yv = np.meshgrid(np.arange(mm), np.arange(nn))
    white_ind = x[-1]==1
    black_ind = x[-1]==-1
    plt.plot(Xv[white_ind],Yv[white_ind],'rs',linewidth=1, markersize=2)
    plt.plot(Xv[black_ind],Yv[black_ind],'bs',linewidth=1, markersize=2)
    plt.show()
    
if __name__ == '__main__':
    J = -1.0
    B = 0
    kB = 1.38064852e-23
    temp = 10e23
    calc = Ising_2D_Calc(temp, J, B)
    sample_length = 500
#    x0 = np.random.choice([-1,1],size=(N,N))
    x0 = np.ones((N,N)) # block initialization (test with negative J)
#    x0 = repmat(np.array([[1,-1],[-1,1]]),N//2,N//2) # scattered initialization (test with positive J)
    Xv, Yv = np.meshgrid(np.arange(N), np.arange(N))
    
    mcmc = MH_Sampler(h=None, T=flip_one_bit, N=sample_length, x0=x0, calc=calc)
    
    energy = []
    magnet = []
    for n in range(sample_length):
        mcmc.step()
        energy.append(calc.potential(mcmc.x))
        magnet.append(calc.avg_magnet(mcmc.x, N))
#        white_ind = mcmc.X[-1]==1
#        black_ind = mcmc.X[-1]==-1
#        plt.plot(Xv[white_ind],Yv[white_ind],'rs',linewidth=1, markersize=2)
#        plt.plot(Xv[black_ind],Yv[black_ind],'bs',linewidth=1, markersize=2)
#        plt.show()
    plt.plot(np.arange(sample_length), energy)
    plt.title('Energy')
    plt.show()
    plt.plot(np.arange(sample_length), magnet)
    plt.title('Average Magnetisation')
    plt.show()
    mcmc.run()
    plot_config(mcmc.X[-100])
    plot_config(mcmc.X[-1])

    
    
    

