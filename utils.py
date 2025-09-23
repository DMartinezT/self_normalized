import numpy as np


class NoiseSampler:
    def __init__(self, type_, radius=1, a=None, b=None):
        self.type_ = type_
        if self.type_ == "uniform":
            self.radius = radius
            self.var = radius**2/ 12
        if self.type_ == "beta":
            self.a = a
            self.b = b
            self.radius = radius
            self.var = (a * b) / (a + b)**2 / (a + b + 1) * radius**2                 

    def __call__(self, size):
        if self.type_ == "uniform":
            return np.random.uniform(low=0, high=self.radius, size=size) - self.radius/2
        elif self.type_ == "beta":
            return ( np.random.beta(self.a, self.b, size) -  self.a / (self.a + self.b) ) * self.radius
        else:
            raise ValueError("Invalid noise type")
    
    def get_variance(self):
        return self.var
    

def psi_e(lam):
    return -lam - np.log(1 - lam)

def psi_g(lam):
    return lam**2 / 2  


class UnivariateEmpiricalBernsteinRadius:
    def __init__(self, max_rounds, B, alpha = 0.05, c1 = 0.5, c2 = 0.25):
        self.max_rounds = max_rounds
        self.B = B
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2

        self.t = 0
        self.sumvarhat = self.c2 * self.B**2
        self.aux = np.sqrt((2*self.B)**2*2*np.log(1 / self.alpha))
        self.part1 = 0
        self.part2 = (2*self.B)*np.log(1/self.alpha)
        self.sum_lambdas = 0
        self.summuhat = self.c1
        self.sum_bernstein_center = 0
        

    def __call__(self, X):


        

        self.t += 1
        radius = (X - self.summuhat/(self.t+2))**2
        self.summuhat += X
        self.sumvarhat += radius
        
        self.lambda_b = np.minimum(self.aux / np.sqrt(self.sumvarhat * np.log(self.t+2)), self.c1) 
        self.sum_lambdas += self.lambda_b
        print(f"lambda_b: {self.lambda_b}")
        # Center
        self.sum_bernstein_center += self.lambda_b * X
        bernstein_center = self.sum_bernstein_center / self.sum_lambdas
        # Radius
        self.part1 += radius*psi_e(self.lambda_b) / (2*self.B)
        bernstein_radius = (self.part1+self.part2)/self.sum_lambdas
        

        return bernstein_center + bernstein_radius
