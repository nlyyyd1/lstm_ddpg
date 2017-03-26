#! /usr/bin/env python
#coding=utf-8
import numpy as np
import numpy.random as nr

class OUNoise:
    def __init__(self,action_dimension,mu=0,theta=0.15,sigma=0.3):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension)*self.mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dimension)*self.mu
        
    def set_sigma(sigma):
        self.sigma = sigma
        
    def noise(self):
        x = self.state
        dx = self.theta*(self.mu-x)+self.sigma*nr.randn(len(x))
        self.state = x+dx
        return self.state
    
if __name__ == '__main__':
    ou = OUNoise(2)
    states = []
    for i in range(10000):
        states.append(ou.noise())
    
    np.savetxt('Noise.txt',np.array(states),newline='\n')
        