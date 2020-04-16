# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:07:05 2020

@author: Jinghong Chen
"""

"""
It is very easy to make coding error in MCMC. So it is necessary that one has ways to check that the MCMC algorithm is correctly implemented. This is simply done by using the algorithm to solve a problem with known answer. In this context, a 2D Ising Model is a suitable 'verification problem'

The Ising model is a statistical mechanics model, typically used to simulate magnetic systems. It consists of a lattice with discrete value σ_j∈{−1,1}  assigned to each point, representing the spin of local eddy current. The Hamiltonian of the system given a configuration of spins is
H=−∑24_(<i,j>)^ ▒〖J_ij σ_i σ_j 〗−B∑24_j^ ▒σ_j 
where the first sum is defined over the immediate neighboring pairs (for d dimension there are 2d of them) 
If J_ij  is positive, the pairs tend to align thus magnetised, i.e., the material is ferromagnetic.  If negative the material is antiferromagnetic. In the obsence of external magnetic field, B=0, the Hamiltonian reduces to
H=−∑24_(<i,j>)^ ▒〖J_ij σ_i σ_j 〗
We use a N×N matrix X with entry either -1 or 1 to represent the magnet configuration. We are interested in drawing samples from the distribution π(X)∝exp⁡(−E/(k_B T)), where
 E=∑24_(<i,j>)^ ▒〖J_ij σ_i σ_j 〗  is the total energy of the configuration, k_B  the boltzmaan constant and T the temperature in Kelvin. 

To verify that the sampling is reasonable, we can check 
	• 1) Whether the equilibrium state responds to change in J 
	• 2) Whether the average energy ⟨E⟩=⟨−∑24_(<i,j>)^ ▒〖J_ij σ_i σ_j 〗⟩  and the average magnetisation per unit spin ⟨M⟩=1/N^2  ∑24_(<i,j>)^ ▒〖σ_i σ_j 〗  converge and
	• 3) whether the sequence of samples looks like a magnetizing/demagnetizing process

"""
import numpy as np

class Ising_2D_Calc:
   
    def __init__(self, temp, J, B):
        """
        @parameters
        temp: temperature in Kelvin
        """
        self.temp = temp
        self.J = J
        self.B = B
        self.kB = 1.38064852e-23
        
    def potential(self, x):
        """
        h(x)=1/(k_B T) ∑24_i^d▒〖J x_i x_(i+1) 〗  and π(X)∝exp⁡(−E/(k_B T))
        """
        nn, mm = x.shape
        sum1 = 0
        sum2 = 0
        for i in range(nn-1):
            sum1 += np.dot(x[i],x[i+1])
        for j in range(mm-1):
            sum2 += np.dot(x[:,j],x[:,j+1])
        return -2 * self.J * (sum1+sum2)
#        return -2 * (1/(self.kB*self.temp) * self.J) * (sum1+sum2)
        
    def avg_magnet(self, x, N):
        """
        calculate the average magnetisation of the current configuration
        """
        return np.sum(x)/N**2

if __name__ == '__main__':
    ising_calc = Ising_2D_Calc(1, 1, 0)
    x = np.eye(3)
    x[0,1] = 1
    print(ising_calc.potential(x))