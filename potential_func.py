import numpy as np
from numpy.matlib import repmat
import timeit

cost_mat = np.loadtxt("data/london_n/cost_mat.txt")
orig = np.loadtxt("data/london_n/P.txt")
xd = np.loadtxt("data/london_n/xd0.txt")
nn, mm = np.shape(cost_mat)
theta = np.zeros(5)

class Calc:
    def __init__(self, theta, orig, cost_mat):
        self.alpha = theta[0]
        self.beta = theta[1]
        self.delta = theta[2]
        self.gamma = theta[3]
        self.kappa = theta[4]
        self.expsum = None
        self.nn, self.mm = cost_mat.shape
        self.cost_mat = cost_mat
        self.orig = orig
        self.xx = None
    
    def _update_expsum(self, xx):
        if self.expsum is None or self.xx is None or (self.xx!=xx).any():
            self.expsum = np.sum(np.exp(repmat(self.alpha*xx,nn,1) - self.beta*self.cost_mat), axis=1)
            self.xx = xx.copy()
    
    def update_theta(self, theta):
        self.alpha = theta[0]
        self.beta = theta[1]
        self.delta = theta[2]
        self.gamma = theta[3]
        self.kappa = theta[4]
        if not self.xx is None:
            self._update_expsum(self.xx)
        
    def potential(self, xx):
        self._update_expsum(xx)
        v_utility = np.sum(orig*np.log(self.expsum)) * (-self.alpha)**-1
        try:
            v_cost = self.kappa*np.sum(np.exp(xx))
        except FloatingPointError:
            print("overflowing. x is:")
            for ele in xx:
                print(f'{ele}, ')

        v_addition = self.delta*np.sum(xx)
        # print("v_utility = ",v_utility)
        return self.gamma*(v_utility + v_cost - v_addition)

    def grad(self, xx):
        self._update_expsum(xx)
        grad = np.zeros(self.mm)
        grad = -np.sum(repmat(self.orig[:,None],1,mm)*np.exp((repmat(self.alpha*xx,self.nn,1)-self.beta*self.cost_mat))/repmat(self.expsum[:,None],1,mm),axis=0)
        grad += self.kappa*np.exp(xx)
        grad -= self.delta
        return self.gamma*grad
    
    def hessian(self, xx):
        self._update_expsum(xx)
        A = self.alpha * repmat(self.orig,self.mm,1) * np.exp(repmat(self.alpha*xx,self.nn,1)-self.beta*self.cost_mat).T/repmat(self.expsum,self.mm,1)
        B = np.exp(repmat(self.alpha*xx,self.nn,1) - self.beta*self.cost_mat)/repmat(self.expsum[:,None],1,self.mm)
        C = np.diag(np.sum(-A, axis=1)) + np.diag(self.kappa*np.exp(xx))
        
                
        return self.gamma*(np.dot(A,B) + C)
    
def pot_value(xx, calc):
    """
    wrapper
    """
    return (calc.potential(xx), calc.grad(xx))
