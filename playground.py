# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:24:55 2020

@author: asus
"""

import numpy as np
from numpy.matlib import repmat
import os
import sys
import platform
import ctypes
from numpy.ctypeslib import ndpointer
import timeit
from potential_func import Calc

#
#class Calc:
#    def __init__(self, theta, orig, cost_mat):
#        self.alpha = theta[0]
#        self.beta = theta[1]
#        self.delta = theta[2]
#        self.gamma = theta[3]
#        self.kappa = theta[4]
#        self.expsum = None
#        self.nn, self.mm = cost_mat.shape
#        self.cost_mat = cost_mat
#        self.orig = orig
#        self.xx = 0
#
#    def potential(self, xx):
#        if self.expsum is None or (self.xx!=xx).any():
##            print('expsum is updated')
#            self.expsum = np.sum(np.exp(repmat(self.alpha*xx,nn,1) - self.beta*self.cost_mat), axis=1)
#            self.xx = xx.copy()
#
#        v_utility = np.sum(orig*np.log(self.expsum)) * (-self.alpha)**-1
#        try:
#            v_cost = self.kappa*np.sum(np.exp(xx))
#        except FloatingPointError:
#            print("overflowing. x is:")
#            for ele in xx:
#                print(f'{ele}, ')
#
#        v_addition = self.delta*np.sum(xx)
#        return self.gamma*(v_utility + v_cost - v_addition)
#
#    def grad(self, xx):
#        grad = np.zeros(self.mm)
#        grad = -np.sum(repmat(self.orig[:,None],1,mm)*np.exp((repmat(self.alpha*xx,self.nn,1)-self.beta*self.cost_mat))/repmat(self.expsum[:,None],1,mm),axis=0)
#        grad += self.kappa*np.exp(xx)
#        grad -= self.delta
#        return self.gamma*grad

if __name__ == '__main__':
    cost_mat = np.loadtxt("data/london_n/cost_mat.txt")
    orig = np.loadtxt("data/london_n/P.txt")
    xd = np.loadtxt("data/london_n/xd0.txt")
    nn, mm = np.shape(cost_mat)
    theta = np.array([2., 0.5*0.7e7, .3/mm, 100., 1.3])
    
    calc = Calc(theta, orig, cost_mat)
    pot = calc.potential(xd)
    gra = calc.grad(xd)
    print(f'potential at xd: {pot}')
    print(f'gradian at grad: {gra}')
    
    xd1 = xd - gra*10e-2
    new_pot = calc.potential(xd1)
    print(f'following gradian new potential is {new_pot}')
#    start = timeit.default_timer()
#    pot = calc.potential(xd)
#    gra = calc.grad(xd)
#    hess = calc.hessian(xd)
#    end = timeit.default_timer()
#    print(f'potential at xd: {pot}')
#    print(f'gradian at xd: {gra}')   
#    print(f'hessian at xd: {hess}')
#    print(f'Taking {end-start} seconds')
