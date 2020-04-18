# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:43:52 2020

@author: Jinghong Chen
"""

import numpy as np
from numpy.matlib import repmat
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
        c_length = self.length
        tic = timeit.default_timer()
        while self.length < c_length+nn:
            x_new = self.T(self.x)
            delta_h = self.calc.potential(x_new)-self.calc.potential(self.x) if self.calc else self.h(x_new, self.h_args) - self.h(self.x, self.h_args)
#            delta_h = self.calc.efficient_potential(x_new)-self.calc.efficient_potential(self.x)
            transition_prob = np.min([np.exp(-delta_h),1])
#            print(transition_prob)
            self.x = x_new if np.random.uniform() <= transition_prob else self.x
            self.X[self.length] = self.x
            self.length += 1
            
            if self.length % 1000 == 0:
                toc = timeit.default_timer()
#                print(f"runing 1000 iterations takes {toc-tic} seconds")
                tic = timeit.default_timer()
    
    def step(self):
        self.run(1)
        
class PT_Sampler:
    '''
    Parallel Tempering
    We augment the space to the product space X_1×X_2×…×X_I  , were the X_i  are identical copies of X. Suppose (x_1,x_2,…,x_I)∈ X_1×X_2×…×X_I. For the famility of distributions Π={π_i (x), i∈I}, we define a joint probability distribution on the product space as
    π_pt (x_1,…,x_I)=∏24_(i∈I)▒〖π_i (x_i ) 〗
    and run parallel MCMC chains on all of the X_i. An "index swapping" operation is conducted in place of the temperature transition in ST.
    
    M1. Let the current state be (x_1,x_2,…,x_I), we draw u~Uniform[0,1]
    M2. If u≤α_0, we conduct parallel step; That is, we update every 〖x_i^ 〗^((t))  to 〖x_i^ 〗^((t+1))  via their respective MCMC scheme
    M3. If u≥ α_0, we conduct the swapping step; That is, we randomly choose a neighboring pair, say i and i+1, and propose "swapping" x_i^((t))  and x_(i+1 )^((t)). Accept with the probability
    min⁡{1,(π_i (x_(i+1)^((t) ) ) π_(i+1) (x_i^((t) ) ))/( π_i (x_i^((t) ) ) π_(i+1) (x_(i+1)^((t) ) ) )}
    '''
    
    def __init__(self, h, T, N, x0, temps, swap_p, calc=None):
        """
        @parameters 
        h: potential function π(X)∝exp⁡(−h), takes temperature as argument
        T: transition function, T(x) returns x', the new state proposal
        N: the maximum length of the Markov Chain
        x0: the initial state of the Markov Chains in the expanded space
        calc: If specified, calc.potential(x, temp) should return h for calculation
        """
        self.temps = temps # temperature levels
        self.swap_p = swap_p
        self.d = temps.size # number of parallel chains
        if x0.shape[0] != self.d:
            self.x0 = repmat(x0,self.d,1)
            self.x0 = np.reshape(self.x0, (self.d,-1))
        self.h = h
        self.T = T
        self.N = N
        self.length = 0 # current length of the chain
        self.x = self.x0.copy() # current configuration
        self.X = np.empty((self.N, *self.x.shape)) # all samples
        self.calc = calc # associated calculator
        self.acs = 0
        self.swaps = 0
        self.ac_rate = 0
        self.swap_rate = 0
        
    def run(self, n=None):
        """
        @parameter
        n: the number of time to extend the Markov Chain 
        """
        nn = n or self.N-self.length
        c_length = self.length
        tic = timeit.default_timer()
        while self.length < c_length+nn and self.length < self.N:
            if np.random.uniform() <= self.swap_p:
            # Conduct parallel step. I.e., update w.r.t individual Markov Chain
                x_new = np.empty_like(self.x)
                for i in range(self.d):  
                    x_new[i] = self.T(self.x[i])
                    delta_h = self.calc.potential(x_new[i],self.temps[i])-self.calc.potential(self.x[i],self.temps[i]) if self.calc else self.h(x_new[i], self.temps[i]) - self.h(self.x[i], self.temps[i])
                    transition_prob = np.min([np.exp(-delta_h),1])
                    if np.random.uniform() <= transition_prob:
                        self.x[i] = x_new[i] 
                        self.acs += 1
                    else:
                        pass
#                    self.x[i] = x_new[i] if np.random.uniform() <= transition_prob else self.x[i]
                    
                
            else:
            # Conduct swapping step
                j = np.random.randint(self.d-1)
                # note that self.calc.potential() gives log(pi)
                prob_num = self.calc.potential(self.x[j],self.temps[j+1]) + self.calc.potential(self.x[j+1],self.temps[j])
                prob_den = self.calc.potential(self.x[j],self.temps[j]) + self.calc.potential(self.x[j+1],self.temps[j+1])
                transition_prob = np.min([1,np.exp(prob_num-prob_den)])
                if np.random.uniform() <= transition_prob:
                    self.x[j], self.x[j+1] = self.x[j+1], self.x[j]
                    self.swaps += 1
                else:
                    pass
                
            self.X[self.length] = self.x
            self.length += 1
        
            if self.length % 1000 == 0:
                    toc = timeit.default_timer()
    #                print(f"runing 1000 iterations takes {toc-tic} seconds")
                    tic = timeit.default_timer()
                    
        self.ac_rate = self.acs/self.length
        self.swap_rate = self.swaps/self.length

    def step(self):
        self.run(1)