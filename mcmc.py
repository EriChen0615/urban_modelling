# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:43:52 2020

@author: Jinghong Chen
"""

import numpy as np
import timeit

class MH_Sampler:
    """
    Monte Carlo Markov Chain Metropolis-Hasting Framework Sampler
    
    The Metropolis algorithm
    M1： Propose a random "unbiased pertubation" of the current state x_t  so as to genrate a new configuration x^′; Calculate the change Δh=h(x^′ )−h(x_t )
    M2:    Generate a random number U ~ Uniform[0,1]. Let x_(t+1)=x^′  if
    U≤π(x^′ )/π(x_t ) =exp⁡(−Δh)
    and let x_(t+1)=x_t  otherwise
    
    Note that the pertubation rule is restricted to be symmetric, i.e.,
     T(x,x^′ )=T(x^′,x) for Metropolis Sampler
    where T is a transition function

    """
    def __init__(self, h, T, N, x0, calc=None, h_args=None, T_args=None):
        """
        @parameters 
        h: potential function π(X)∝exp⁡(−h)
        T: transition function, T(x) returns x', the new state proposal
        N: the maximum length of the Markov Chain
        d: shape of the state variable x
        x0: the initial state of the Markov Chain
        calc: If specified, calc.potential(x) should return h for calculation
        """
        self.x0 = x0
        self.h = h
        self.T = T
        self.N = N
        self.length = 0 # current length of the chain
        self.x = x0 # current configuration
        self.X = np.empty((self.N, *self.x.shape)) # all samples
        self.calc = calc # associated calculator
        self.T_args = T_args
        self.h_args = h_args
        
    def run(self, n=None):
        """
        @parameter
        n: the number of time to extend the Markov Chain 
        """
        nn = n or self.N-self.length
        tic = timeit.default_timer()
        for i in range(nn):
            x_new = self.T(self.x)
            delta_h = self.calc.potential(x_new)-self.calc.potential(self.x) if self.calc else self.h(x_new, self.h_args) - self.h(self.x, self.h_args)
#            delta_h = self.calc.efficient_potential(x_new)-self.calc.efficient_potential(self.x)
            transition_prob = np.min([np.exp(-delta_h),1])
#            print(transition_prob)
            self.x = x_new if np.random.uniform() <= transition_prob else self.x
            self.X[self.length+i] = self.x
            self.length += 1
            
            if i % 1000 == 0:
                toc = timeit.default_timer()
                print(f"runing 1000 iterations takes {toc-tic} seconds")
                tic = timeit.default_timer()
    
    def step(self):
        self.run(1)