# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:40:58 2020

@author: Jinghong Chen
"""
from ising import Ising_2D_Calc
from mcmc import MH_Sampler
import numpy as np
import matplotlib.pyplot as plt

def random_flip(x, N):
    new_x = x.copy()
    new_x[np.random.randint(N, size=10), np.random.randint(N, size=10)] *= -1
    return new_x

def flip_one_bit(x):
    new_x = x.copy()
    flip_num = 5
    new_x[np.random.randint(50,size=flip_num), np.random.randint(50,size=flip_num)] *= -1
    return new_x

if __name__ == '__main__':
    calc = Ising_2D_Calc(273, -1, 0)
    N = 50
    sample_length = 1000
#    x0 = np.random.choice([-1,1],size=(N,N))
    x0 = np.ones((N,N))
    Xv, Yv = np.meshgrid(np.arange(N), np.arange(N))
    
    mcmc = MH_Sampler(h=None, T=flip_one_bit, d=(N,N), N=sample_length, x0=x0, calc=calc)
    energy = []
    magnet = []
    for n in range(sample_length):
        mcmc.run(1)
        energy.append(calc.potential(mcmc.x))
        magnet.append(calc.avg_magnet(mcmc.x, N))
        white_ind = mcmc.X[-1]==1
        black_ind = mcmc.X[-1]==-1
        plt.plot(Xv[white_ind],Yv[white_ind],'gs',linewidth=1)
        plt.plot(Xv[black_ind],Yv[black_ind],'bs',linewidth=1)
#        plt.contour(Xv,Yv,mcmc.X[-1], levels=[-0.99,0.99],colors=['b','g'])
        plt.show()
    plt.plot(np.arange(sample_length), energy)
    plt.title('Energy')
    plt.show()
    plt.plot(np.arange(sample_length), magnet)
    plt.title('Average Magnetisation')
    plt.show()
#    mcmc.run()
    
    
    

