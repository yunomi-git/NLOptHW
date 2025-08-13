import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#################
# generating data

f = lambda x: x**2 / 3 - 3 * x * np.sin(x-2) - np.cos(x)
# f = lambda x: x*np.cos(x) / 3
m = lambda x: np.zeros_like(x)

#################

def K_rbf(x, y, k, sigma):
    # return k * np.exp(- np.linalg.norm(x - y, axis=-1) ** 2 / (2.0 * sigma))
    return k * np.exp(- 0.5 * ((x - y) / sigma)**2)

def pdf(x, mu, sigma):
    # print(1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- 0.5 * ((x - mu) / sigma)**2), x - mu, sigma)
    return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- 0.5 * ((x - mu) / sigma)**2)

def repeat_first_dim(x, num_repeats):
    return np.stack(np.repeat(x[np.newaxis, :], axis=0, repeats=num_repeats), axis=0)

def compute_post(grid_x, m, observed_x, observed_f, k_rbf, sigma_rbf):
    # grid_x: the set of points we want to compute the posterior distribution
    # m: m(x) returns the prior guess
    # observed_x: observed points
    # observed_f: function value of observed points
    # k_rbf, sigma_rbf: parameters of RBF kernel

    # return mu_x, sigma_x: mean and variance of grid_x

    #############################
    # write your code here
    mu_x = np.zeros(len(grid_x))
    sigma_x = np.zeros(len(grid_x))
    num_observations = len(observed_x)

    # first construct cov matrix
    observations_stacked = repeat_first_dim(observed_x, num_observations)
    observations_transposed = np.swapaxes(observations_stacked, 0, 1)
    cov_matrix = K_rbf(observations_stacked, observations_transposed, k=k_rbf, sigma=sigma_rbf)

    # Then construct the error
    error = observed_f - m(observed_x)

    cov_error_product = np.linalg.solve(cov_matrix, error)

    # Calculate each value in grid
    for i in range(len(grid_x)):
        x = grid_x[i]
        input_cov = K_rbf(x, observed_x, k=k_rbf, sigma=sigma_rbf)

        mu_x[i] = m(x) + input_cov.T @ cov_error_product
        sigma_x[i] = K_rbf(x, x, k=k_rbf, sigma=sigma_rbf) - input_cov.T @ np.linalg.solve(cov_matrix, input_cov)
        sigma_x[i] = np.sqrt(sigma_x[i])

    sigma_x[sigma_x < 0] = 1e-10
    ##############################

    return mu_x, sigma_x

if __name__=="__main__":
    a = compute_post(grid_x=np.array([-1, 1]),
                     m=m,
                     observed_x=np.array([1, 2, 3]),
                     observed_f=np.array([2, 3, 4]),
                     k_rbf=1,
                     sigma_rbf=1)
    print(a)