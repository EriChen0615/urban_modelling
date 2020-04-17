# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 00:56:49 2020

@author: asus
"""

from potential_func import Calc
from mcmc import MH_Sampler:
import numpy as np
import matplotlib.pyplot as plt

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
    
if __name__ == '__main__':
    cost_mat = np.loadtxt("data/london_n/cost_mat.txt")
    orig = np.loadtxt("data/london_n/P.txt")
    xd = np.loadtxt("data/london_n/xd0.txt")
    nn, mm = np.shape(cost_mat)
    theta = np.array([2., 0.5*0.7e7, .3/mm, 100., 1.3])
    
    calc = Calc(theta, orig, cost_mat)

    sample_length = 30000
    
    mcmc = MH_Sampler(h=None, T=flip_one_bit, d=mm, N=sample_length, x0=xd, calc=calc)
    
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
    
    
    

