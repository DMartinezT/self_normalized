import numpy as np
from scipy.special import gamma, gammaincc, gammainc
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import NoiseSampler, UnivariateEmpiricalBernsteinRadius




class KernelizedUCB:
    def __init__(self, noise_subgaussian_parameter, noise_bound, noise_variance, kernel, kernel_bound, regression_function_bound, delta=0.05, rho=1.0, rho_mixture=1, max_total_number_points=1000):
        self.kernel = kernel
        self.kernel_bound = kernel_bound
        self.regression_function_bound = regression_function_bound
        self.noise_subgaussian_parameter = noise_subgaussian_parameter
        self.noise_variance = noise_variance
        self.delta = delta
        self.rho = rho
        self.rho_mixture = rho_mixture
        self.beta_det = None
        self.beta1_det = None
        self.beta_pinelis = None
        self.beta1_pinelis = None
        self.beta_empirical_pinelis = None
        self.beta1_empirical_pinelis = None

        self.sum_gt2 = 0
        self.gt2_det = None

        self.X = None
        self.y = None
        self.K_inv = None
        self.noise_bound = noise_bound

        self.eb = UnivariateEmpiricalBernsteinRadius(max_rounds=max_total_number_points, B=noise_bound, alpha=delta/2, c1=0.5, c2=0.25)
        
    def fit(self, X, y, n_new_points):

        if self.y is not None:
            K_s = self.kernel(X[-n_new_points], self.X)
            mu = K_s @ self.K_inv @ self.y
        else:
            mu = np.zeros(y[-n_new_points:].shape)
            
        empirical_variance = self.eb((y[-n_new_points:] - mu)**2)
        print("Empirical variance: ", empirical_variance)

        self.X = X
        self.y = y
        K = self.kernel(X, X) + self.rho * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)

        self.gt2_det = np.log(np.sqrt(np.linalg.det(self.kernel(X, X) / self.rho + np.eye(len(X)))) / self.delta)
        self.beta1_det = self.noise_subgaussian_parameter*np.sqrt(2 * self.gt2_det)
        self.beta_det = np.sqrt(self.rho)*self.regression_function_bound + self.beta1_det

        K_new_points = self.kernel(X[-n_new_points:], X)
        K_only_new_points = self.kernel(X[-n_new_points:], X[-n_new_points:])
        g_t2 = (1/self.rho)* (K_only_new_points - K_new_points @ self.K_inv @ K_new_points.T )
        g_t2 = np.sum(g_t2)
        self.sum_gt2 += g_t2
        self.beta1_pinelis = self.gamma_poisson_mixture_bound(v=self.sum_gt2*self.noise_variance, 
                                                              rho=self.rho_mixture, c=self.noise_bound*self.kernel_bound/(self.rho+self.kernel_bound), 
                                                              l0=2, delta=self.delta)
        
        self.beta_pinelis = np.sqrt(self.rho)*self.regression_function_bound + self.beta1_pinelis


    def predict(self, X):
        K_s = self.kernel(X, self.X)
        K_ss = self.kernel(X, X)
        mu = K_s @ self.K_inv @ self.y
        sigma = np.sqrt(np.diag(K_ss - K_s @ self.K_inv @ K_s.T))
        return mu, sigma

    def select_action(self, X):
        mu, sigma = self.predict(X)
        ucb_values = mu + self.beta_pinelis * sigma
        return np.argmax(ucb_values)
        
    def visualize_selection(self, X, true_regression_function = None):
        mu, sigma = self.predict(X)
        ucb_values_det = mu + self.beta_det * sigma
        ucb_values_pinelis = mu + self.beta_pinelis * sigma

        plt.figure(figsize=(10, 6))
        plt.plot(X, mu, 'r', label='Mean')
        plt.fill_between(X.ravel(), mu - self.beta_det * sigma, mu + self.beta_det * sigma, alpha=0.2, label='Confidence Interval sub-Gaussian')
        plt.plot(X, ucb_values_det, 'g', label='UCB Values sub-Gaussian')
        plt.plot(X, ucb_values_pinelis, 'orange', label='UCB Values Pinelis')
        plt.scatter(self.X, self.y, c='b', label='Training Points')
        if true_regression_function is not None:
            plt.plot(X, true_regression_function(X), 'k', label='True Function')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('UCB Selection Process')
        plt.legend()
        # plt.show()
        # Save the plot to a file (optional)
        plt.savefig("../plots/visualize_selection.png", dpi=300)     

    
    def visualize_variance(self, X, true_regression_function = None):
        mu, sigma = self.predict(X)

        plt.figure(figsize=(10, 6))
        plt.plot(X, mu, 'r', label='Mean')
        plt.fill_between(X.ravel(), mu - sigma, mu + sigma, alpha=0.2, label='mu +- sigma')
        # plt.plot(X, sigma, 'g', label='Sigma')
        plt.scatter(self.X, self.y, c='b', label='Training Points')
        if true_regression_function is not None:
            plt.plot(X, true_regression_function(X), 'k', label='True Function')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Sigma')
        plt.legend()
        # plt.show()
        # Save the plot to a file (optional)
        plt.savefig("../plots/sigma.png", dpi=300)  

    

    def gamma_exponential_mixture(self, s, v, rho, c): ## Proposition 9 Howard et al. 2021
        part1 = (rho / c**2)**(rho/c**2) / (gamma(rho/c**2) * gammainc(rho/c**2, rho/c**2))
        part2 = gamma((v + rho)/c**2)  * gammainc((v + rho)/c**2, (c*s + v + rho)/c**2) / ((c*s + v + rho)/c**2)**((v + rho)/c**2)
        part3 = np.exp((c*s + v) / c**2)
        m = part1 * part2 * part3
        return m
    
    def gamma_exponential_mixture_bound(self, v, rho, c, l0, delta):
        equation_to_solve = lambda s: self.gamma_exponential_mixture(s, v, rho, c) - l0 / delta
        s_solution = brentq(equation_to_solve, 0, 10)  # Adjust the interval [0, 100] as needed
        return s_solution

    def gamma_poisson_mixture(self, s, v, rho, c): ## Proposition 10 Howard et al. 2021
        part1 = (rho / c**2)**(rho/c**2) / (gamma(rho/c**2) * gammaincc(rho/c**2, rho/c**2))
        part2 = gamma((c*s + v + rho)/c**2)  * gammaincc((c*s + v + rho)/c**2, (v + rho)/c**2) / ((v + rho)/c**2)**((c*s + v + rho)/c**2)
        part3 = np.exp(v / c**2)
        m = part1 * part2 * part3
        return m

    def gamma_poisson_mixture_bound(self, v, rho, c, l0, delta):
        equation_to_solve = lambda s: self.gamma_poisson_mixture(s, v, rho, c) - l0 / delta
        s_solution = brentq(equation_to_solve, 0, 100)  # Adjust the interval [0, 100] as needed
        return s_solution

        
def plot_function(function, xmin, ymin, num_points=100):
    x = np.linspace(xmin, ymin, num_points)
    y = function(x)
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('True Regression Function')
    plt.show()





# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    kernel = RBF(length_scale=.01)
    kernel_bound = 1
    D = 5
    rho = .05
    radius_noise = 1
    rho_mixture = .01 * radius_noise**2
    delta = 0.1
    noise_type = "beta"
    max_rounds = 5000
    initial_number_of_points = 1
    


    if noise_type == "uniform":
        noise_sampler = NoiseSampler(type_="uniform", radius=radius_noise)
    elif noise_type == "beta":
        a = 50
        b = 50
        noise_sampler = NoiseSampler(type_="beta", a=a, b=b, radius=radius_noise)
        
   
    noise_subgaussian_parameter = radius_noise**2/4
    noise_variance = noise_sampler.get_variance()
    ucb = KernelizedUCB(noise_subgaussian_parameter=noise_subgaussian_parameter, noise_bound=radius_noise, noise_variance=noise_variance, 
                        kernel=kernel, kernel_bound=kernel_bound, regression_function_bound=D, rho=rho, delta=delta, rho_mixture=rho_mixture, 
                        max_total_number_points=max_rounds+initial_number_of_points)

    n_fixed_points = 50
    fixed_points = np.vstack((np.random.rand(n_fixed_points // 2 - 2, 1) / 10 + 0.1, np.random.rand(n_fixed_points // 2 + 2, 1) / 10 + 0.8))
    weights = np.random.randn(n_fixed_points) 
    weights = np.ones(n_fixed_points)
    weights *= D / np.sum(weights)

    # Generate some sample data
    X_train = np.random.rand(initial_number_of_points, 1)
    true_regression_function = lambda x: np.sum(weights * kernel(x, fixed_points), axis=1)
    y_train = true_regression_function(X_train) + noise_sampler(size=initial_number_of_points)
    # Fit the model with the updated data
    ucb.fit(X_train, y_train, n_new_points=initial_number_of_points)

    # plot_function(true_regression_function, 0, 1)
    # y_train = np.sin(2 * np.pi * X_train).ravel() + np.random.randn(10) * 0.1
    
    
    for _ in tqdm(range(max_rounds)):

        # Generate some test data
        X_test = np.linspace(0, 1, 100).reshape(-1, 1)
        action = ucb.select_action(X_test)

        # print(f"Round {_ + 1}: Selected action: {action}")

        # Generate a new sample point
        X_new = X_test[action].reshape(1, 1)
        y_new = true_regression_function(X_new) + noise_sampler(size=1)

        # Update the training data
        X_train = np.vstack((X_train, X_new))
        y_train = np.append(y_train, y_new)

        # Fit the model with the updated data
        ucb.fit(X_train, y_train, n_new_points=1)

    ucb.visualize_selection(X_test, true_regression_function)
    # ucb.visualize_variance(X_test, true_regression_function)

    print(f"pinelis beta1: {ucb.beta1_pinelis}")
    print(f"det beta1: {ucb.beta1_det}")
    print(f"pinelis sum_gt2: {ucb.sum_gt2*ucb.noise_variance*np.log(2/delta)}")
    print(f"det sum_gt2: {ucb.noise_subgaussian_parameter**2*ucb.gt2_det}")


    print(f"Noise variance: {noise_variance}")
    print(f"Noise subgaussian parameter: {noise_subgaussian_parameter}")

    


    

    
    

   
    