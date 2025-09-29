import numpy as np

def psi_e(lam):
    return -lam - np.log(1 - lam)
    
def psi_p(lam):
    return np.exp(lam) - lam - 1


class EBUB:
    def __init__(self, max_rounds, alpha = 0.05, c1 = 0.5, c2 = 0.25**2, c3=0.25, c4=0.5, CS=False):
        self.max_rounds = max_rounds
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4

        self.sumvarhat = self.c2
        self.summuhat = self.c3
        self.tilde_summuhat = self.c4

        self.aux = np.sqrt(2*np.log(1 / self.alpha))
        self.part1 = 0
        self.part2 = np.log(1/self.alpha)

        self.sum_lambdas = 0
        self.sum_bernstein_center = 0
        self.CS = CS
        self.t = 0
        
        

    def __call__(self, Y, muhat):

        X = (Y - muhat)**2
        radius = (X - self.summuhat/(self.t+1))**2

        if self.CS:
            self.lambda_b = np.minimum(self.aux / np.sqrt(self.sumvarhat * np.log(self.t+2)), self.c1) 
        else:
            self.lambda_b = np.minimum(self.aux / np.sqrt(self.sumvarhat * self.max_rounds / (self.t+1)), self.c1)
        self.sum_lambdas += self.lambda_b
        
        # Center
        self.sum_bernstein_center += self.lambda_b * X
        self.bernstein_center = self.sum_bernstein_center / self.sum_lambdas
        # Radius
        self.part1 += radius*psi_e(self.lambda_b)
        self.bernstein_radius = (self.part1+self.part2)/self.sum_lambdas

        # Update auxiliary variables
        self.summuhat += X
        self.sumvarhat += radius
        self.t += 1

    def get_center_plus_radius(self):
        return np.sqrt(self.bernstein_center + self.bernstein_radius)

    def get_center(self):
        return np.sqrt(self.bernstein_center)
    