import numpy as np
from scipy.special import gamma, gammaincc, gammainc
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import NoiseSampler, UnivariateEmpiricalBernsteinRadius
from ebub import EBUB




class KernelizedUCB:
    def __init__(self, noise_subgaussian_parameter, noise_bound, noise_variance, kernel, kernel_bound, regression_function_bound, delta=0.05, rho=1.0, rho_mixture=1, max_total_number_points=1000,
                 noise_type="uniform"):
        self.kernel = kernel
        self.kernel_bound = kernel_bound
        self.regression_function_bound = regression_function_bound
        self.noise_subgaussian_parameter = noise_subgaussian_parameter
        self.noise_variance = noise_variance
        self.delta = delta
        self.delta1 = delta / 2
        self.delta2 = delta / 2
        self.rho = rho
        self.rho_mixture = rho_mixture
        self.beta_det = None
        self.beta1_det = None
        self.beta_pinelis = None
        self.beta1_pinelis = None
        self.beta_empirical_pinelis = None
        self.beta1_empirical_pinelis = None
        self.noise_type = noise_type

        self.sum_gt2 = 0
        self.gt2_det = None

        self.X = None
        self.y = None
        self.K_inv = None
        self.noise_bound = noise_bound

        #self.eb = UnivariateEmpiricalBernsteinRadius(max_rounds=max_total_number_points, B=noise_bound, alpha=delta/2, c1=0.5, c2=0.25)
        self.eb = EBUB(max_rounds=max_total_number_points, alpha = self.delta1, c1 = 0.5, c2 = 0.25**2, c3=0.25, c4=0.5, CS=False)


    def fit(self, X, y, n_new_points):

        if self.y is not None:
            K_s = self.kernel(X[-n_new_points], self.X)
            mu = K_s @ self.K_inv @ self.y
        else:
            mu = np.zeros(y[-n_new_points:].shape)
            
        #empirical_variance = self.eb((y[-n_new_points:] - mu)**2)
        for yy, m in zip(y[-n_new_points:], mu):
            self.eb(yy, m)
        

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
        # self.beta1_pinelis = self.gamma_poisson_mixture_bound(v=self.sum_gt2*self.noise_variance, 
        #                                                       rho=self.rho_mixture, c=self.noise_bound*self.kernel_bound/(self.rho+self.kernel_bound), 
        #                                                       l0=2, delta=self.delta)
        
        # self.beta_pinelis = np.sqrt(self.rho)*self.regression_function_bound + self.beta1_pinelis


    def predict(self, X):
        K_s = self.kernel(X, self.X)
        K_ss = self.kernel(X, X)
        mu = K_s @ self.K_inv @ self.y
        sigma = np.sqrt(np.diag(K_ss - K_s @ self.K_inv @ K_s.T))
        return mu, sigma

    def select_action(self, X):
        mu, sigma = self.predict(X)
        #ucb_values = mu + self.beta_pinelis * sigma
        ucb_values = mu + self.beta_det * sigma
        return np.argmax(ucb_values)

    def visualize_selection(self, X, true_regression_function = None, ax_index = None, ax=None):
        mu, sigma = self.predict(X)
        ucb_values_det = mu + self.beta_det * sigma

        self.beta1_pinelis = self.gamma_poisson_mixture_bound(v=self.sum_gt2*self.noise_variance, 
                                                              rho=self.rho_mixture, c=self.noise_bound*self.kernel_bound/(self.rho+self.kernel_bound), 
                                                              l0=2, delta=self.delta)
        
        self.beta_pinelis = np.sqrt(self.rho)*self.regression_function_bound + self.beta1_pinelis
        ucb_values_pinelis = mu + self.beta_pinelis * sigma


        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        #plt.fill_between(X.ravel(), mu - self.beta_det * sigma, mu + self.beta_det * sigma, alpha=0.2, label='Confidence Interval sub-Gaussian')
        ax.plot(X, ucb_values_det, 'g', label='Sub-Gaussian')
        ax.plot(X, ucb_values_pinelis, 'orange', label='Mixed Bennett')


        empirical_variance = self.eb.get_center_plus_radius()
        print("Empirical variance: ", empirical_variance)
        self.beta1_pinelis_empirical = self.gamma_poisson_mixture_bound(v=self.sum_gt2*self.noise_variance, 
                                                              rho=self.rho_mixture, c=self.noise_bound*self.kernel_bound/(self.rho+self.kernel_bound), 
                                                              l0=2, delta=self.delta2)
        
        self.beta_pinelis_empirical = np.sqrt(self.rho)*self.regression_function_bound + self.beta1_pinelis_empirical
        ucb_values_pinelis_empirical = mu + self.beta_pinelis_empirical * sigma
        ax.plot(X, ucb_values_pinelis_empirical, 'purple', label='Empirical Mixed Bennett')

        ax.scatter(self.X, self.y, c='b', label='Training Points')
        ax.plot(X, mu, 'r', label='Estimated mean')
        if true_regression_function is not None:
            ax.plot(X, true_regression_function(X), 'k', label='True regression function')

        # Uncomment for recent contributions
        '''
        self.metelli_first_part = np.sqrt(73 *np.log(np.linalg.det(self.kernel(X, X) / self.rho + np.eye(len(X)))) ) + np.sqrt(3)
        rho_t = np.log (1 + (self.noise_bound ** 2) * self.kernel_bound**2 / (self.rho * self.noise_variance))
        rho_t *= np.log(8 * (self.noise_bound ** 2) * self.kernel_bound**2 * (len(X)**3) / (self.rho * self.noise_variance))
        rho_t = np.ceil(rho_t)
        rho_t = np.max([0, rho_t])
        self.metelli_first_part *= np.sqrt(np.log(np.pi**2 * (rho_t + 1)**2 / (3 * self.delta)))
        self.metelli_second_part = (3*self.noise_bound*self.kernel_bound) / np.sqrt(self.rho * self.noise_variance)
        self.metelli_second_part *= np.log(np.pi**2 * (rho_t + 1)**2 / (3 * self.delta))
        self.metelli1_det = self.metelli_first_part + self.metelli_second_part
        self.metelli1_det *= np.sqrt(self.noise_variance)
        self.metelli_det = np.sqrt(self.rho)*self.regression_function_bound + self.metelli1_det
        ucb_values_metelli = mu + self.metelli_det * sigma
        ax.plot(X, ucb_values_metelli, color='pink', label='Metelli et al.')

        rho_akhavan = self.noise_variance * self.rho
        rho_akhavan = self.rho
        gamma_akhavan = np.log(np.linalg.det(self.kernel(X, X) / rho_akhavan + np.eye(len(X)))) / 2
        akhavan_first_part = np.sqrt(3*rho_akhavan)/2 + 2*np.sqrt(3)/np.sqrt(rho_akhavan)*(9*gamma_akhavan + np.log(2/delta)) 
        akhavan_second_part = np.sqrt(6*(gamma_akhavan + np.log(2/delta)))
        akhavan1_det = akhavan_first_part + akhavan_second_part
        akhavan1_det *= np.sqrt(self.noise_variance)
        akhavan_det = np.sqrt(self.rho)*self.regression_function_bound + akhavan1_det
        ucb_values_akhavan = mu + akhavan_det * sigma
        ax.plot(X, ucb_values_akhavan, 'c', label='Akhavan et al.')

        print(f"gamma: {np.log(np.linalg.det(self.kernel(X, X) / self.rho + np.eye(len(X))))}")
        print(f"sqrt gamma: {np.sqrt(np.log(np.linalg.det(self.kernel(X, X) / self.rho + np.eye(len(X))))) }")
        print(f"gamma akhavan: {gamma_akhavan}")
        print(f"noise bound: {self.noise_bound}")
        '''



        ax.set_xlabel(r'$\tilde{X}$')
        ax.set_ylabel('Y')
        
        axis_to_title = ["(I)", "(II)", "(III)", "(IV)"]
        plot_title = axis_to_title[ax_index-1]
        ax.set_title(plot_title if plot_title is not None else 'UCB Selection')

        #if ax_index == 1:
        ax.legend(loc="upper center")
        # plt.show()
        # Save the plot to a file (optional)
        if ax is None:  
            plt.savefig(f"../plots/visualize_selection_{self.noise_type}_{self.noise_bound}.png", dpi=300)
        

    
    def visualize_variance(self, X, true_regression_function = None):
        mu, sigma = self.predict(X)

        plt.figure(figsize=(10, 6))
        plt.plot(X, mu, 'r', label='Mean')
        plt.fill_between(X.ravel(), mu - sigma, mu + sigma, alpha=0.2, label='mu +- sigma')
        # plt.plot(X, sigma, 'g', label='Sigma')
        plt.scatter(self.X, self.y, c='b', label='Training Points')
        if true_regression_function is not None:
            plt.plot(X, true_regression_function(X), 'k', label='True Function')

        plt.xlabel(r'$\tilde{X}$')
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
    radius_noise_list = [2]
    delta = 0.1
    noise_type_list = ["uniform", "beta55", "beta2020", "beta5050"] #uniform
    max_rounds = 500
    initial_number_of_points = 1

    fig, axes = plt.subplots(len(noise_type_list), len(radius_noise_list), figsize=(14, 5) )  # (14, 10)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10) )  # (14, 5)
    if len(noise_type_list) == 1:
        axes = np.array([axes])  # make it 2D for consistency
    axes = axes.flatten()  # flatten to a 1D array for easy indexing

    ax_index = 0
    for noise_type in noise_type_list:
        for radius_noise in radius_noise_list:
            ax = axes[ax_index]
            ax_index += 1

            print(f"Noise type: {noise_type}, Noise radius: {radius_noise}")

            rho_mixture = .01 * radius_noise**2
            rho_mixture = 1
    


            if noise_type == "uniform":
                noise_sampler = NoiseSampler(type_="uniform", radius=radius_noise)
            elif noise_type == "beta":
                a = 10
                b = 10
                noise_sampler = NoiseSampler(type_="beta", a=a, b=b, radius=radius_noise)
            elif noise_type == "beta55":
                a = 5
                b = 5
                noise_sampler = NoiseSampler(type_="beta", a=a, b=b, radius=radius_noise)
            elif noise_type == "beta2020":
                a = 20
                b = 20
                noise_sampler = NoiseSampler(type_="beta", a=a, b=b, radius=radius_noise)
            elif noise_type == "beta5050":
                a = 50
                b = 50
                noise_sampler = NoiseSampler(type_="beta", a=a, b=b, radius=radius_noise)
                
        
            noise_subgaussian_parameter = radius_noise**2/4
            noise_variance = noise_sampler.get_variance()
            ucb = KernelizedUCB(noise_subgaussian_parameter=noise_subgaussian_parameter, noise_bound=radius_noise, noise_variance=noise_variance, 
                                kernel=kernel, kernel_bound=kernel_bound, regression_function_bound=D, rho=rho, delta=delta, rho_mixture=rho_mixture, 
                                max_total_number_points=max_rounds+initial_number_of_points, noise_type=noise_type)

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

            ucb.visualize_selection(X_test, true_regression_function, ax_index=ax_index, ax=ax) #f"{noise_type} noise with bound {radius_noise}"
            # ucb.visualize_variance(X_test, true_regression_function)

            print(f"pinelis beta1: {ucb.beta1_pinelis}")
            print(f"det beta1: {ucb.beta1_det}")
            print(f"pinelis sum_gt2: {ucb.sum_gt2*ucb.noise_variance*np.log(2/delta)}")
            print(f"det sum_gt2: {ucb.noise_subgaussian_parameter**2*ucb.gt2_det}")


            print(f"Noise variance: {noise_variance}")
            print(f"Noise subgaussian parameter: {noise_subgaussian_parameter}")
    #plt.tight_layout()
    plt.savefig(f"../plots/visualize_selection_all.png", dpi=300, bbox_inches='tight')

    


    

    
    

   
    